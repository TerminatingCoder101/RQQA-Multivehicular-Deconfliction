using OSQP
using Plots
using LinearAlgebra
using SparseArrays

# Vehicle Dynamics Model from Pachter 2024
# state = [x, y, psi, v] (position, heading, and forward speed)
# control = [a, w] (forward acceleration and angular velocity)

function vehicle_dynamics(state, control)
    x, y, psi, v = state
    a, w = control

    x_dot = v * cos(psi)
    y_dot = v * sin(psi)
    psi_dot = w
    v_dot = a

    return [x_dot, y_dot, psi_dot, v_dot]
end

const DT = 0.1 
const T_MAX = 25.0
const N_STEPS = Int(T_MAX / DT)

const R_AGENT = 0.5
const D_MIN_TOTAL = 1.5 * R_AGENT + 1 
const D_THREAT = 4.0

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 3.0
const K2 = 2.0

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const V_MAX_3 = 2.5 # Speed for the second pursuer
const A_MIN, A_MAX = -1.0, 1.0
const W_MIN, W_MAX = -pi, pi

function calc_intercept_pt(state_evader, state_pursuer, v_pursuer_max)
    p1 = state_evader[1:2]
    psi1 = state_evader[3]
    v1 = state_evader[4]

    p2 = state_pursuer[1:2]
    v2 = v_pursuer_max

    delta_p = p1 - p2
    v1_vec = v1 * [cos(psi1), sin(psi1)]

    a = dot(v1_vec, v1_vec) - v2^2
    b = 2 * dot(delta_p, v1_vec)
    c = dot(delta_p, delta_p)

    discriminant = b^2 - 4*a*c
    if discriminant < 0
        return p1 + v1_vec * 0.1
    end

    t1 = (-b + sqrt(discriminant)) / (2a)
    t2 = (-b - sqrt(discriminant)) / (2a)

    t = -1.0
    if t1 > 0 && t2 > 0
        t = min(t1, t2)
    elseif t1 > 0
        t = t1
    elseif t2 > 0
        t = t2
    else
        return p1 + v1_vec * 0.1
    end

    return p1 + v1_vec * t
end
    

function run_simulation()
    # --- Vehicle Initialization ---
    # Vehicle 1 (Evader)
    state1 = [-10.0, 1.0, 0.0, 0.0]
    goal1 = [10.0, 1.0]
    hist_state1 = [state1]; hist_control1 = [[0.0, 0.0]]

    # Vehicle 2 (Pursuer 1)
    state2 = [5.0, -1.0, pi, V_MAX_2]
    hist_state2 = [state2]; hist_control2 = [[0.0, 0.0]]

    # Vehicle 3 (Pursuer 2)
    state3 = [5.0, 3.0, -pi/2, V_MAX_3]
    hist_state3 = [state3]; hist_control3 = [[0.0, 0.0]]

    # History for plotting
    interceptionHist1 = []; interceptionHist2 = []
    hist_h_12 = Float64[]; hist_h_13 = Float64[]; hist_h_23 = Float64[]

    println("Starting 2 vs 1 simulation...")

    for i in 1:N_STEPS
        if norm(state1[1:2] - goal1) < 0.3
            println("Evader reached goal at step $i. Stopping simulation.")
            break
        end

        # --- Nominal Controllers ---
        # Evader Controller
        error1 = goal1 - state1[1:2]
        psi_error1 = atan(sin(atan(error1[2], error1[1]) - state1[3]), cos(atan(error1[2], error1[1]) - state1[3]))
        v_turn_based = V_MAX_1 * (1.0 - 0.1 * abs(psi_error1) / pi)
        dist_to_pursuer = min(norm(state1[1:2] - state2[1:2]), norm(state1[1:2] - state3[1:2]))
        v_evasive = dist_to_pursuer < D_THREAT ? V_MAX_1 * (dist_to_pursuer / D_THREAT) : V_MAX_1
        v_desired = min(v_turn_based, v_evasive)
        a_n1 = 0.5 * (v_desired - state1[4])
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([a_n1, w_n1], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # Pursuer 1 Controller
        interceptionPt1 = calc_intercept_pt(state1, state2, V_MAX_2)
        push!(interceptionHist1, interceptionPt1)
        error2 = interceptionPt1 - state2[1:2]
        psi_error2 = atan(sin(atan(error2[2], error2[1]) - state2[3]), cos(atan(error2[2], error2[1]) - state2[3]))
        v_desired_2 = V_MAX_2 * (1.0 - 0.7 * abs(psi_error2) / pi)
        a_n2 = 0.5 * (v_desired_2 - state2[4])
        w_n2 = 2.0 * psi_error2
        u_n2 = clamp.([a_n2, w_n2], [A_MIN, W_MIN], [A_MAX, W_MAX])
        
        # Pursuer 2 Controller
        interceptionPt2 = calc_intercept_pt(state1, state3, V_MAX_3)
        push!(interceptionHist2, interceptionPt2)
        error3 = interceptionPt2 - state3[1:2]
        psi_error3 = atan(sin(atan(error3[2], error3[1]) - state3[3]), cos(atan(error3[2], error3[1]) - state3[3]))
        v_desired_3 = V_MAX_3 * (1.0 - 0.7 * abs(psi_error3) / pi)
        a_n3 = 0.5 * (v_desired_3 - state3[4])
        w_n3 = 2.0 * psi_error3
        u_n3 = clamp.([a_n3, w_n3], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # Helper function to compute CBF constraints for any pair of vehicles
        function get_cbf_pair_constraints(s_i, s_j)
            pi = s_i[1:2]; psii = s_i[3]; vi = s_i[4]
            pj = s_j[1:2]; psij = s_j[3]; vj = s_j[4]
            delta_p = pi - pj
            vi_vec = vi * [cos(psii), sin(psii)]
            vj_vec = vj * [cos(psij), sin(psij)]
            delta_v = vi_vec - vj_vec
            h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
            h_dot = 2 * dot(delta_p, delta_v)
            Lfh = 2*dot(delta_v, delta_v) + K1*h_dot + K2*h
            g_ai = 2 * dot(delta_p, [cos(psii), sin(psii)])
            g_wi = 2 * vi * dot(delta_p, [-sin(psii), cos(psii)])
            g_aj = -2 * dot(delta_p, [cos(psij), sin(psij)])
            g_wj = -2 * vj * dot(delta_p, [-sin(psij), cos(psij)])
            return h, h_dot, Lfh, g_ai, g_wi, g_aj, g_wj
        end

        h12, h_dot12, Lfh12, g_a1_12, g_w1_12, g_a2_12, g_w2_12 = get_cbf_pair_constraints(state1, state2)
        h13, h_dot13, Lfh13, g_a1_13, g_w1_13, g_a3_13, g_w3_13 = get_cbf_pair_constraints(state1, state3)
        h23, h_dot23, Lfh23, g_a2_23, g_w2_23, g_a3_23, g_w3_23 = get_cbf_pair_constraints(state2, state3)
        
        push!(hist_h_12, h12); push!(hist_h_13, h13); push!(hist_h_23, h23)

        # Build the QP matrices
        A_cbf = [ g_a1_12  g_w1_12  g_a2_12  g_w2_12    0        0;
                  g_a1_13  g_w1_13    0        0      g_a3_13  g_w3_13;
                    0        0      g_a2_23  g_w2_23  g_a3_23  g_w3_23 ]
        
        b_cbf_lower = [-Lfh12; -Lfh13; -Lfh23]
        
        H = diagm([20.0, 0.1, 20.0, 0.1, 20.0, 0.1])
        P = sparse(H * 2.0)
        q = -2.0 * H * [u_n1; u_n2; u_n3]
        A = sparse([A_cbf; I(6)])
        l = [b_cbf_lower; A_MIN; W_MIN; A_MIN; W_MIN; A_MIN; W_MIN]
        u = [Inf; Inf; Inf; A_MAX; W_MAX; A_MAX; W_MAX; A_MAX; W_MAX]

        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        u_safe = results.info.status == :Solved ? results.x : [u_n1; u_n2; u_n3]
        
        u1_safe = u_safe[1:2]; u2_safe = u_safe[3:4]; u3_safe = u_safe[5:6]

        # Update States
        state1 = state1 + vehicle_dynamics(state1, u1_safe) * DT
        state1[4] = clamp(state1[4], 0.0, V_MAX_1)
        state2 = state2 + vehicle_dynamics(state2, u2_safe) * DT 
        state2[4] = clamp(state2[4], 0.0, V_MAX_2)
        state3 = state3 + vehicle_dynamics(state3, u3_safe) * DT 
        state3[4] = clamp(state3[4], 0.0, V_MAX_3)

        # Store History
        push!(hist_state1, state1); push!(hist_state2, state2); push!(hist_state3, state3)
        push!(hist_control1, u1_safe); push!(hist_control2, u2_safe); push!(hist_control3, u3_safe)
    end

    println("Simulation finished.")
    return hist_state1, hist_state2, hist_state3, hist_control1, hist_control2, hist_control3, hist_h_12, hist_h_13, hist_h_23, interceptionHist1, interceptionHist2
end


function plot_and_animate(hist_state1, hist_state2, hist_state3, hist_control1, hist_control2, hist_control3, hist_h_12, hist_h_13, hist_h_23, interceptionHist1, interceptionHist2)
    println("Generating plots and animation...")
    
    num_steps_run = length(hist_state1) - 1
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)
    time_axis_cbf = 0:DT:((length(hist_h_12)-1) * DT)

    x1=[s[1] for s in hist_state1]; y1=[s[2] for s in hist_state1]; v1=[s[4] for s in hist_state1]; w1=[c[2] for c in hist_control1[2:end]]
    x2=[s[1] for s in hist_state2]; y2=[s[2] for s in hist_state2]; v2=[s[4] for s in hist_state2]; w2=[c[2] for c in hist_control2[2:end]]
    x3=[s[1] for s in hist_state3]; y3=[s[2] for s in hist_state3]; v3=[s[4] for s in hist_state3]; w3=[c[2] for c in hist_control3[2:end]]

    goal1_pos = [10.0, 1.0]
    theta = 0:0.1:(2*pi+0.1)

    p_traj = plot(x1, y1, label="V1 (Evader)", lw=2, aspect_ratio=:equal, xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectories")
    plot!(p_traj, x2, y2, label="V2 (Pursuer 1)", lw=2)
    plot!(p_traj, x3, y3, label="V3 (Pursuer 2)", lw=2)
    scatter!(p_traj, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)

    p_vel = plot(time_axis_states, v1, label="V1", lw=2, title="Velocity Profiles", xlabel="Time [s]", ylabel="Velocity [m/s]")
    plot!(p_vel, time_axis_states, v2, label="V2", lw=2)
    plot!(p_vel, time_axis_states, v3, label="V3", lw=2)

    p_rot = plot(time_axis_controls, w1, label="V1", lw=2, title="Rotational Speeds", xlabel="Time [s]", ylabel="ω [rad/s]")
    plot!(p_rot, time_axis_controls, w2, label="V2", lw=2)
    plot!(p_rot, time_axis_controls, w3, label="V3", lw=2)

    p_h = plot(time_axis_cbf, hist_h_12, label="h_12", lw=2, title="CBF Values (Safety)", xlabel="Time [s]", ylabel="h(x)")
    plot!(p_h, time_axis_cbf, hist_h_13, label="h_13", lw=2)
    plot!(p_h, time_axis_cbf, hist_h_23, label="h_23", lw=2)
    hline!(p_h, [0], linestyle=:dash, color=:red, label="Boundary (h=0)")

    static_plot = plot(p_traj, p_vel, p_rot, p_h, layout=(2,2), size=(1200, 800))
    savefig(static_plot, "cbf_analysis_2v1.png")
    println("Analysis plots saved to cbf_analysis_2v1.png")

    anim = @animate for i in 1:lastindex(interceptionHist1)
        p_anim = plot(x1[1:i], y1[1:i], label="V1 Path", lw=2, aspect_ratio=:equal, xlims=(-12, 12), ylims=(-6, 6),
             xlabel="x [m]", ylabel="y [m]", title="2 vs 1 Pursuit-Evasion (Frame $i)")
        plot!(p_anim, x2[1:i], y2[1:i], label="V2 Path", lw=2)
        plot!(p_anim, x3[1:i], y3[1:i], label="V3 Path", lw=2)
        
        plot!(p_anim, x1[i].+R_AGENT.*cos.(theta), y1[i].+R_AGENT.*sin.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="V1", c=:blue)
        plot!(p_anim, x2[i].+R_AGENT.*cos.(theta), y2[i].+R_AGENT.*sin.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="V2", c=:orange)
        plot!(p_anim, x3[i].+R_AGENT.*cos.(theta), y3[i].+R_AGENT.*sin.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="V3", c=:purple)
        
        scatter!(p_anim, [interceptionHist1[i][1]], [interceptionHist1[i][2]], label="Intcpt 1", marker=:star5, markersize=6, color=:red)
        scatter!(p_anim, [interceptionHist2[i][1]], [interceptionHist2[i][2]], label="Intcpt 2", marker=:star5, markersize=6, color=:pink)
        scatter!(p_anim, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)
    end

    gif(anim, "cbf_simulation_2v1.gif", fps = 15)
    println("Animation saved to cbf_simulation_2v1.gif")
end

# Run Simulation and Generate GIF
hist_state1, hist_state2, hist_state3, hist_control1, hist_control2, hist_control3, hist_h_12, hist_h_13, hist_h_23, interceptionHist1, interceptionHist2 = run_simulation()
plot_and_animate(hist_state1, hist_state2, hist_state3, hist_control1, hist_control2, hist_control3, hist_h_12, hist_h_13, hist_h_23, interceptionHist1, interceptionHist2)

println("Script finished. Press Enter to exit...")
readline()

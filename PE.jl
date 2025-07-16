using OSQP
using Plots
using LinearAlgebra
using SparseArrays

# Vehicle Dynamics Model: Second-order unicycle model
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

# Simulation parameters
const DT = 0.1 # Time step [s]
const T_MAX = 25.0
const N_STEPS = Int(T_MAX / DT)

# Agent Properties
const R_AGENT = 0.5
const D_MIN_TOTAL = 1.5 * R_AGENT + 1 
const D_THREAT = 5.0 # Distance at which evader starts evasive maneuvers

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 3.0
const K2 = 2.0

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const A_MIN, A_MAX = -1.0, 1.0
const W_MIN, W_MAX = -pi, pi

"""
Calculates the intercept point by solving for the time-to-intercept.
This is the validated function from our test script.
"""
function calc_intercept_pt(state_evader, state_pursuer)
    p1 = state_evader[1:2]
    psi1 = state_evader[3]
    v1 = state_evader[4]

    p2 = state_pursuer[1:2]
    v2 = V_MAX_2 # Pursuer will travel at max speed to intercept

    delta_p = p1 - p2
    v1_vec = v1 * [cos(psi1), sin(psi1)]

    # Quadratic equation coefficients for time-to-intercept 't'
    # (v1^2 - v2^2)t^2 + (2*dot(delta_p, v1_vec))t + norm(delta_p)^2 = 0
    a = dot(v1_vec, v1_vec) - v2^2
    b = 2 * dot(delta_p, v1_vec)
    c = dot(delta_p, delta_p)

    discriminant = b^2 - 4*a*c
    if discriminant < 0
        # No real solution, pursuer cannot intercept. Aim ahead.
        return p1 + v1_vec * 0.1
    end

    # Solve for time t
    t1 = (-b + sqrt(discriminant)) / (2a)
    t2 = (-b - sqrt(discriminant)) / (2a)

    # Get the smallest positive time
    t = -1.0
    if t1 > 0 && t2 > 0
        t = min(t1, t2)
    elseif t1 > 0
        t = t1
    elseif t2 > 0
        t = t2
    else
        # No positive time solution, aim ahead
        return p1 + v1_vec * 0.1
    end

    return p1 + v1_vec * t
end
    

function run_simulation()
    # Initialize history arrays
    interceptionHist = []
    hist_h = Float64[]
    hist_h_dot = Float64[]

    # Vehicle 1 (Evader): Starts on the left, goal is on the right
    state1 = [-10.0, 1.0, 0.0, 0.0]
    goal1 = [10.0, 1.0]
    hist_state1 = [state1]
    hist_control1 = [[0.0, 0.0]]

    # Vehicle 2 (Pursuer): Starts on the right, goal is Vehicle 1
    state2 = [5.0, -1.0, pi, V_MAX_2]
    hist_state2 = [state2]
    hist_control2 = [[0.0, 0.0]]

    println("Starting simulation...")

    for i in 1:N_STEPS
        if norm(state1[1:2] - goal1) < 0.3
            println("Evader reached goal at step $i. Stopping simulation.")
            break
        end

        # --- Nominal Controller for Vehicle 1 (Evader) ---
        error1 = goal1 - state1[1:2]
        dist_to_goal = norm(error1)
        angle_to_goal1 = atan(error1[2], error1[1])
        psi_error1 = atan(sin(angle_to_goal1 - state1[3]), cos(angle_to_goal1 - state1[3]))
        v_goal_based = 0.9 * dist_to_goal
        v_turn_based = V_MAX_1 * (1.0 - 0.5 * abs(psi_error1) / pi)
        dist_to_pursuer = norm(state1[1:2] - state2[1:2])
        v_evasive = dist_to_pursuer < D_THREAT ? V_MAX_1 * (dist_to_pursuer / D_THREAT) : V_MAX_1
        v_desired = min(v_goal_based, v_turn_based, v_evasive)
        a_n1 = 0.5 * (v_desired - state1[4])
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([a_n1, w_n1], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # --- Nominal Controller for Vehicle 2 (Pursuer) ---
        interceptionPt = calc_intercept_pt(state1, state2)
        push!(interceptionHist, interceptionPt)
        goal2 = interceptionPt
        error2 = goal2 - state2[1:2]
        angle_to_goal2 = atan(error2[2], error2[1])
        psi_error2 = atan(sin(angle_to_goal2 - state2[3]), cos(angle_to_goal2 - state2[3]))
        
        # FIX: Implement "Brake to Turn" logic
        w_n2 = 8.0 * psi_error2 # Command an aggressive turn
        
        # If the turn is sharp (e.g., > 45 degrees), command max braking.
        # Otherwise, accelerate towards the target speed.
        if abs(psi_error2) > pi / 4.0
            a_n2 = A_MIN
        else
            v_desired_2 = V_MAX_2 * (1.0 - 0.7 * abs(psi_error2) / pi)
            a_n2 = 0.5 * (v_desired_2 - state2[4])
        end

        u_n2 = clamp.([a_n2, w_n2], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # --- Second-Order CBF-QP Safety Filter ---
        p1 = state1[1:2]; psi1 = state1[3]; v1 = state1[4]
        p2 = state2[1:2]; psi2 = state2[3]; v2 = state2[4]
        delta_p = p1 - p2
        v1_vec = v1 * [cos(psi1), sin(psi1)]
        v2_vec = v2 * [cos(psi2), sin(psi2)]
        delta_v = v1_vec - v2_vec
        h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
        h_dot = 2 * dot(delta_p, delta_v)
        push!(hist_h, h)
        push!(hist_h_dot, h_dot)
        Lfh = 2*dot(delta_v, delta_v) + K1*h_dot + K2*h
        g_a1 = 2 * dot(delta_p, [cos(psi1), sin(psi1)])
        g_w1 = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])
        g_a2 = -2 * dot(delta_p, [cos(psi2), sin(psi2)])
        g_w2 = -2 * v2 * dot(delta_p, [-sin(psi2), cos(psi2)])
        A_cbf = [g_a1 g_w1 g_a2 g_w2] 
        b_cbf_lower = -Lfh
        H = diagm([20.0, 0.1, 20.0, 0.1])
        P = sparse(H * 2.0)
        q = -2.0 * H * [u_n1; u_n2]
        A = sparse([A_cbf; I(4)])
        l = [b_cbf_lower; A_MIN; W_MIN; A_MIN; W_MIN]
        u = [Inf; A_MAX; W_MAX; A_MAX; W_MAX]
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        u_safe = results.info.status == :Solved ? results.x : [u_n1; u_n2]
        u1_safe = u_safe[1:2]
        u2_safe = u_safe[3:4]
        state1 = state1 + vehicle_dynamics(state1, u1_safe) * DT
        state1[4] = clamp(state1[4], 0.0, V_MAX_1)
        state2 = state2 + vehicle_dynamics(state2, u2_safe) * DT 
        state2[4] = clamp(state2[4], 0.0, V_MAX_2)
        push!(hist_state1, copy(state1))
        push!(hist_state2, copy(state2))
        push!(hist_control1, u1_safe)
        push!(hist_control2, u2_safe)
    end

    println("Simulation finished.")
    return hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist
end


function plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist)
    println("Generating plots and animation...")
    
    num_steps_run = length(hist_state1) - 1
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)
    time_axis_cbf = 0:DT:((length(hist_h)-1) * DT)

    x1_hist = [s[1] for s in hist_state1]; y1_hist = [s[2] for s in hist_state1]
    v1_hist = [s[4] for s in hist_state1]; w1_hist = [c[2] for c in hist_control1[2:end]]
    a1_hist = [c[1] for c in hist_control1]
    
    x2_hist = [s[1] for s in hist_state2]; y2_hist = [s[2] for s in hist_state2]
    v2_hist = [s[4] for s in hist_state2]; w2_hist = [c[2] for c in hist_control2[2:end]]
    a2_hist = [c[1] for c in hist_control2]

    goal1_pos = [10.0, 1.0]
    theta = 0:0.1:(2*pi+0.1)

    p_traj = plot(x1_hist, y1_hist, label="V1 Path (Evader)", lw=2, aspect_ratio=:equal,
                  xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectories")
    plot!(p_traj, x2_hist, y2_hist, label="V2 Path (Pursuer)", lw=2)
    scatter!(p_traj, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)

    p_vel = plot(time_axis_states, v1_hist, label="V1 (Evader)", lw=2, title="Velocity Profiles", xlabel="Time [s]", ylabel="Velocity [m/s]")
    plot!(p_vel, time_axis_states, v2_hist, label="V2 (Pursuer)", lw=2)
    hline!(p_vel, [V_MAX_1, V_MAX_2], linestyle=:dash, color=:red, label="Vel. Bounds")

    p_accel = plot(time_axis_states, a1_hist, label="A1 (Evader)", lw=2, title="Control Acceleration", xlabel="Time [s]", ylabel="Acceleration [m/s^2]")
    plot!(p_accel, time_axis_states, a2_hist, label="A2 (Pursuer)", lw=2)
    hline!(p_accel, [A_MAX, A_MIN], linestyle=:dash, color=:red, label="Bounds")

    p_rot = plot(time_axis_controls, w1_hist, label="V1 (Evader)", lw=2, title="Rotational Speeds", xlabel="Time [s]", ylabel="ω [rad/s]")
    plot!(p_rot, time_axis_controls, w2_hist, label="V2 (Pursuer)", lw=2)
    hline!(p_rot, [W_MAX, W_MIN], linestyle=:dash, color=:blue, label="Angular Velocity Bounds")

    p_h = plot(time_axis_cbf, hist_h, label="h(x)", lw=2, title="CBF Value (Safety Metric)", xlabel="Time [s]", ylabel="h(x) [DU^2]")
    hline!(p_h, [0], linestyle=:dash, color=:red, label="Boundary (h=0)")

    p_h_dot = plot(time_axis_cbf, hist_h_dot, label="h_dot(x)", lw=2, title="CBF Derivative (Closure Rate)", xlabel="Time [s]", ylabel="h_dot(x) [DU^2/s]")

    static_plot = plot(p_traj, p_vel, p_accel, p_rot, p_h, p_h_dot, layout=(2,3), size=(1800, 1000))
    
    plot_path = "cbf_analysis_plots_pursuit2.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")

    anim = @animate for i in 1:lastindex(interceptionHist)
        p_anim = plot(x1_hist[1:i], y1_hist[1:i], label="V1 Path", lw=2, aspect_ratio=:equal,
             xlims=(-12, 12), ylims=(-6, 6),
             xlabel="x [m]", ylabel="y [m]", title="Pursuit-Evasion with CBF (Frame $i)")
        plot!(p_anim, x2_hist[1:i], y2_hist[1:i], label="V2 Path", lw=2)
        plot!(p_anim, x1_hist[i] .+ R_AGENT .* cos.(theta), y1_hist[i] .+ R_AGENT .* sin.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 1", color=:blue)
        plot!(p_anim, x2_hist[i] .+ R_AGENT .* cos.(theta), y2_hist[i] .+ R_AGENT .* sin.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 2", color=:orange)
        intercept_pt = interceptionHist[i]
        scatter!(p_anim, [intercept_pt[1]], [intercept_pt[2]], label="Intercept Pt.", marker=:star5, markersize=6, color=:red)
        scatter!(p_anim, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)
    end

    gif_path = "cbf_simulation_pursuit2.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end

# Run Simulation and Generate GIF
hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist = run_simulation()
plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist)

println("Script finished. Press Enter to exit...")
readline()

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
const D_THREAT = 3.0

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 3.0
const K2 = 2.0

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const A_MIN, A_MAX = -1.0, 1.0
const W_MIN, W_MAX = -pi, pi

function calc_intercept_pt(state1, state2)
    p1 = state1[1:2]
    psi1 = state1[3]
    v1 = state1[4]

    p2 = state2[1:2]
    v2 = V_MAX_2

    delta_p = p1 - p2
    v1_vec = v1 * [cos(psi1), sin(psi1)]

    # Quadratic equation coefficients for time-to-intercept 't'
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
    interceptionHist = []

    # Vehicle 1 (Evader): Starts on the left, goal is on the right
    state1 = [-10.0, 1.0, 0.0, 0.0]
    goal1 = [10.0, 1.0]
    hist_state1 = [state1]
    hist_control1 = [[0.0, 0.0]]

    # Vehicle 2 (Pursuer): Starts on the right, goal is point of interception
    state2 = [5.0, -1.0, pi, V_MAX_2]
    hist_state2 = [state2]
    hist_control2 = [[0.0, 0.0]]

    println("Starting simulation...")

    hist_h = Float64[]
    hist_h_dot = Float64[]

    # Simulation Loop
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
        v_turn_based = V_MAX_1 * (1.0 - 0.5 * abs(psi_error1) / pi)
        dist_to_pursuer = norm(state1[1:2] - state2[1:2])
        v_evasive = dist_to_pursuer < D_THREAT ? V_MAX_1 * (dist_to_pursuer / D_THREAT) : V_MAX_1
        v_desired = min(v_turn_based, v_evasive, 0.9 * dist_to_goal)
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
        v_desired_2 = V_MAX_2 * (1.0 - 0.7 * abs(psi_error2) / pi)
        a_n2 = 0.5 * (v_desired_2 - state2[4])
        w_n2 = 2.0 * psi_error2
        u_n2 = clamp.([a_n2, w_n2], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # --- Evader's Worst-Case CBF-QP Safety Filter ---
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

        g_a1 = 2 * dot(delta_p, [cos(psi1), sin(psi1)])
        g_w1 = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])

        # Function to calculate the lower bound of the CBF constraint for a given pursuer action
        function get_b_cbf(u_pursuer)
            a2, w2 = u_pursuer
            a2_vec = a2 * [cos(psi2), sin(psi2)] - v2 * w2 * [sin(psi2), -cos(psi2)]
            h_ddot_part = 2*dot(delta_v, delta_v) + 2*dot(delta_p, -a2_vec) + K1*h_dot + K2*h
            return -h_ddot_part
        end

        # Define the 3 worst-case pursuer actions
        u2_left = [a_n2, W_MAX]
        u2_straight = [a_n2, 0.0]
        u2_right = [a_n2, W_MIN]

        # Build the constraint matrix and lower bound vector
        A_cbf = [g_a1 g_w1; g_a1 g_w1; g_a1 g_w1]
        b_cbf_lower = [get_b_cbf(u2_left); get_b_cbf(u2_straight); get_b_cbf(u2_right)]

        # QP for the evader ONLY
        H = diagm([20.0, 0.1])
        P = sparse(H * 2.0)
        q = -2.0 * H * u_n1
        A = sparse([A_cbf; I(2)])
        l = [b_cbf_lower; A_MIN; W_MIN]
        u = [Inf; Inf; Inf; A_MAX; W_MAX]

        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        
        u1_safe = results.info.status == :Solved ? results.x : u_n1
        u2_safe = u_n2 

        # Update States
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

    x1_hist = [s[1] for s in hist_state1]; y1_hist = [s[2] for s in hist_state1]
    v1_hist = [s[4] for s in hist_state1]; w1_hist = [c[2] for c in hist_control1[2:end]]
    a1_hist = [c[1] for c in hist_control1]
    
    x2_hist = [s[1] for s in hist_state2]; y2_hist = [s[2] for s in hist_state2]
    v2_hist = [s[4] for s in hist_state2]; w2_hist = [c[2] for c in hist_control2[2:end]]
    a2_hist = [c[1] for c in hist_control2]

    goal1_pos = [10.0, 1.0]
    theta = 0:0.1:(2*pi+0.1)

    # ... (Static plots can be added here if desired) ...

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

    gif_path = "cbf_simulation_worst_case.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end

# Run Simulation and Generate GIF
hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist = run_simulation()
plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2, hist_h, hist_h_dot, interceptionHist)

println("Script finished. Press Enter to exit...")
readline()

using OSQP
using Plots
using LinearAlgebra
using SparseArrays

#==============================================================================
# SIMULATION PARAMETERS
==============================================================================#

# Simulation parameters
const DT = 0.1 # Time step [s]
const T_MAX = 20.0
const N_STEPS = Int(T_MAX / DT)

# Agent physical properties
const R_AGENT = 0.5 # Radius of agent
const D_MIN_TOTAL = 2.5 * R_AGENT # Minimum safe distance between agent centers

# --- Pursuit Logic ---
const T_PREDICT = 0.1 # Prediction horizon for intercept logic [s]

# --- i-HOCBF Parameters ---
# Based on the paper: https://arxiv.org/pdf/2203.07978
const K1 = 3.0   # Corresponds to alpha_1 in the paper
const K2 = 2.0   # Corresponds to alpha_2 gain in the paper
const K_I = 1.5  # Integral gain for the sigma term

# --- Vehicle Dynamics Control Model ---
# We model the vehicle's linear acceleration as a first-order response
# to a velocity command, a = K_v * (v_ref - v_current).
# This allows us to use [v_ref, w] as controls while maintaining a relative degree of 2.
const K_v = 1.0 # Gain for the velocity controller

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const W_MIN, W_MAX = -pi, pi


#==============================================================================
# SIMULATION
==============================================================================#
function run_simulation()

    # Vehicle 1 (Evader): Starts on the left, goal is on the right
    state1 = [-10.0, 1.0, 0.0, V_MAX_1]
    goal1 = [10.0, 1.0]
    hist_state1 = [state1]
    hist_control1 = [[0.0, 0.0]]

    # Vehicle 2 (Pursuer): Starts on the right, goal is Vehicle 1
    state2 = [5.0, -1.0, pi, V_MAX_2]
    hist_state2 = [state2]
    hist_control2 = [[0.0, 0.0]]
    
    # History for visualization
    hist_intercept_points = []
    
    # Integral CBF state variable
    sigma = 0.0
    hist_cbf = [(h=0.0, h_dot=0.0, sigma=0.0)] # History for h, h_dot, sigma

    println("Starting simulation with Intercept Logic and i-HOCBF...")

    # --- Simulation Loop ---
    for i in 1:N_STEPS
        # Check if the goal has been reached
        if norm(state1[1:2] - goal1) < 0.3
            println("Evader reached goal at step $i. Stopping simulation.")
            break
        end

        # --- Nominal Controllers (Output desired v_ref and w) ---
        # Nominal Controller for Vehicle 1 (Evader)
        error1 = goal1 - state1[1:2]
        angle_to_goal1 = atan(error1[2], error1[1])
        psi_error1 = atan(sin(angle_to_goal1 - state1[3]), cos(angle_to_goal1 - state1[3]))
        v_n1 = V_MAX_1 # Command max speed
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([v_n1, w_n1], [0.0, W_MIN], [V_MAX_1, W_MAX])

        # --- Nominal Controller for Vehicle 2 (Pursuer) with Intercept Logic ---
        # Predict evader's future position based on its current velocity
        p1_current = state1[1:2]
        v1_current_vec = state1[4] * [cos(state1[3]), sin(state1[3])]
        goal2 = p1_current + v1_current_vec * T_PREDICT # The intercept point
        push!(hist_intercept_points, goal2) # Store for visualization

        # The rest of the pursuer's controller uses this new intercept point as the goal
        error2 = goal2 - state2[1:2]
        angle_to_goal2 = atan(error2[2], error2[1])
        psi_error2 = atan(sin(angle_to_goal2 - state2[3]), cos(angle_to_goal2 - state2[3]))
        v_n2 = V_MAX_2 # Command max speed
        w_n2 = 2.0 * psi_error2
        u_n2 = clamp.([v_n2, w_n2], [0.0, W_MIN], [V_MAX_2, W_MAX])


        # --- Integral Higher-Order CBF-QP Safety Filter ---

        # Extract states for clarity
        p1 = state1[1:2]; psi1 = state1[3]; v1 = state1[4]
        p2 = state2[1:2]; psi2 = state2[3]; v2 = state2[4]

        # Calculate relative vectors
        delta_p = p1 - p2
        v1_vec = v1 * [cos(psi1), sin(psi1)]
        v2_vec = v2 * [cos(psi2), sin(psi2)]
        delta_v = v1_vec - v2_vec

        # CBF function (h) and its time derivative (h_dot)
        h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
        h_dot = 2 * dot(delta_p, delta_v) # This is L_f h

        # --- Integral Term Update ---
        psi_1 = h_dot + K1 * h
        sigma_dot = -K_I * sigma - psi_1
        sigma = sigma + sigma_dot * DT # Euler integration

        # --- QP Constraint Formulation (A_cbf * u >= b_cbf_lower) ---
        L_f2_h = 2*dot(delta_v, delta_v) - 2*K_v*v1*dot(delta_p, [cos(psi1),sin(psi1)]) + 2*K_v*v2*dot(delta_p, [cos(psi2),sin(psi2)])
        g_v_ref1 = 2 * K_v * dot(delta_p, [cos(psi1), sin(psi1)])
        g_w1 = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])
        g_v_ref2 = -2 * K_v * dot(delta_p, [cos(psi2), sin(psi2)])
        g_w2 = -2 * v2 * dot(delta_p, [-sin(psi2), cos(psi2)])
        A_cbf = [g_v_ref1 g_w1 g_v_ref2 g_w2] 
        b_cbf_lower = -L_f2_h - (K1 + K2 - 1)*h_dot - (K1*K2 - K1)*h - (K2 - K_I)*sigma

        # --- QP Formulation ---
        H = diagm([1.0, 0.1, 1.0, 0.1])
        P = sparse(H * 2.0)
        q = -2.0 * H * [u_n1; u_n2]
        A = sparse([A_cbf; I(4)])
        l = [b_cbf_lower; 0.0; W_MIN; 0.0; W_MIN]
        u = [Inf; V_MAX_1; W_MAX; V_MAX_2; W_MAX]
        
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        
        u_safe = [u_n1; u_n2] # Default to nominal if solver fails
        if results.info.status == :Solved
            u_safe = results.x
        else
            println("Warning: QP not solved at step $i. Status: $(results.info.status). Using nominal control.")
        end

        u1_safe = u_safe[1:2] # [v_ref1, w1]
        u2_safe = u_safe[3:4] # [v_ref2, w2]

        # --- Update State and Store History ---
        a1 = K_v * (u1_safe[1] - state1[4])
        a2 = K_v * (u2_safe[1] - state2[4])
        state1_dot = [state1[4] * cos(state1[3]), state1[4] * sin(state1[3]), u1_safe[2], a1]
        state2_dot = [state2[4] * cos(state2[3]), state2[4] * sin(state2[3]), u2_safe[2], a2]
        state1 = state1 + state1_dot * DT
        state2 = state2 + state2_dot * DT 
        
        push!(hist_state1, state1)
        push!(hist_state2, state2)
        push!(hist_control1, u1_safe)
        push!(hist_control2, u2_safe)
        push!(hist_cbf, (h=h, h_dot=h_dot, sigma=sigma))
    end

    println("Simulation finished.")
    return hist_state1, hist_state2, hist_control1, hist_control2, hist_cbf, hist_intercept_points
end


#==============================================================================
# PLOTTING AND ANIMATION
==============================================================================#
function plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2, hist_cbf, hist_intercept_points)
    println("Generating plots and animation...")
    
    num_steps_run = length(hist_state1) - 1
    # Create separate time axes for states (N+1 points) and controls (N points)
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)

    x1_hist = [s[1] for s in hist_state1]; y1_hist = [s[2] for s in hist_state1]
    v1_hist = [s[4] for s in hist_state1]
    
    x2_hist = [s[1] for s in hist_state2]; y2_hist = [s[2] for s in hist_state2]
    v2_hist = [s[4] for s in hist_state2]

    # Extract control histories, skipping the initial dummy value
    v_ref1_hist = [c[1] for c in hist_control1[2:end]]
    w1_hist = [c[2] for c in hist_control1[2:end]]
    v_ref2_hist = [c[1] for c in hist_control2[2:end]]
    w2_hist = [c[2] for c in hist_control2[2:end]]

    h_hist = [c.h for c in hist_cbf]; sigma_hist = [c.sigma for c in hist_cbf]

    goal1_pos = [10.0, 1.0]
    theta = 0:0.1:(2*pi+0.1)

    # --- Generate Static Plots for Analysis ---
    p_traj = plot(x1_hist, y1_hist, label="V1 Path (Evader)", lw=2, aspect_ratio=:equal,
                  xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectories")
    plot!(p_traj, x2_hist, y2_hist, label="V2 Path (Pursuer)", lw=2)
    scatter!(p_traj, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)

    p_cbf = plot(time_axis_states, h_hist, label="h(x)", lw=2, title="i-HOCBF State", xlabel="Time [s]", ylabel="Value")
    plot!(p_cbf, time_axis_states, sigma_hist, label="sigma(t)", lw=2)
    hline!(p_cbf, [0], lw=1, ls=:dash, color=:black, label="")

    p_vel = plot(time_axis_states, v1_hist, label="V1 Actual", lw=2, c=:blue, title="Linear Velocity", xlabel="Time [s]", ylabel="Velocity [m/s]")
    plot!(p_vel, time_axis_controls, v_ref1_hist, label="V1 Ref", lw=2, ls=:dash, c=:lightblue)
    plot!(p_vel, time_axis_states, v2_hist, label="V2 Actual", lw=2, c=:red)
    plot!(p_vel, time_axis_controls, v_ref2_hist, label="V2 Ref", lw=2, ls=:dash, c=:orange)
    
    p_w = plot(time_axis_controls, w1_hist, label="V1 (Evader)", lw=2, title="Angular Velocity Control", xlabel="Time [s]", ylabel="w [rad/s]")
    plot!(p_w, time_axis_controls, w2_hist, label="V2 (Pursuer)", lw=2)

    static_plot = plot(p_traj, p_cbf, p_vel, p_w, layout=(2,2), size=(1200, 800), legend=:bottomleft)
    
    plot_path = "ihocbf_intercept_analysis.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")


    # --- Generate Animation ---
    anim = @animate for i in 1:length(hist_intercept_points)
        p_anim = plot(x1_hist[1:i], y1_hist[1:i], label="V1 Path", lw=2, aspect_ratio=:equal,
             xlims=(-12, 12), ylims=(-6, 6),
             xlabel="x [m]", ylabel="y [m]", title="Pursuit-Evasion with Intercept Logic (Frame $i)")
        plot!(p_anim, x2_hist[1:i], y2_hist[1:i], label="V2 Path", lw=2)

        plot!(p_anim, x1_hist[i] .+ R_AGENT .* cos.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 1", color=:blue)
        plot!(p_anim, x2_hist[i] .+ R_AGENT .* cos.(theta), seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 2", color=:orange)
              
        scatter!(p_anim, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)
        
        # Plot the current intercept point
        intercept_pt = hist_intercept_points[i]
        scatter!(p_anim, [intercept_pt[1]], [intercept_pt[2]], label="Intercept Pt.", marker=:star5, markersize=6, color=:magenta)
        
        h_val_current = hist_cbf[i].h
        annotate!(p_anim, -11, 5, text("h(x) = $(round(h_val_current, digits=2))", :left, 10))
    end

    gif_path = "ihocbf_intercept_simulation.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end


#==============================================================================
# MAIN EXECUTION
==============================================================================#
hist_state1, hist_state2, hist_control1, hist_control2, hist_cbf, hist_intercept_points = run_simulation()
plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2, hist_cbf, hist_intercept_points)

println("Script finished. Press Enter to exit...")
readline()

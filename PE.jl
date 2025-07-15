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
const T_MAX = 20.0
const N_STEPS = Int(T_MAX / DT)

# Radius of Agent
const R_AGENT = 0.5

# Min Safe distance
const D_MIN_TOTAL = 1.5 * R_AGENT + 1 

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 3.0
const K2 = 2.0

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const A_MIN, A_MAX = -1.0, 1.0
const W_MIN, W_MAX = -pi, pi


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

    println("Starting simulation...")

    # Simulation Loop
    for i in 1:N_STEPS
        # Check if the goal has been reached or not
        if norm(state1[1:2] - goal1) < 0.3
            println("Evader reached goal at step $i. Stopping simulation.")
            break
        end

        # Nominal Controller for Vehicle 1 (Evader)
        error1 = goal1 - state1[1:2]
        angle_to_goal1 = atan(error1[2], error1[1])
        psi_error1 = atan(sin(angle_to_goal1 - state1[3]), cos(angle_to_goal1 - state1[3]))
        a_n1 = 0.5 * (V_MAX_1 - state1[4])
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([a_n1, w_n1], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # Nominal Controller for Vehicle 2 (Pursuer)
        # The pursuer's goal is always the current position of the evader
        goal2 = state1[1:2]
        error2 = goal2 - state2[1:2]
        angle_to_goal2 = atan(error2[2], error2[1])
        psi_error2 = atan(sin(angle_to_goal2 - state2[3]), cos(angle_to_goal2 - state2[3]))
        a_n2 = 0.5 * (V_MAX_2 - state2[4])
        w_n2 = 2.0 * psi_error2
        u_n2 = clamp.([a_n2, w_n2], [A_MIN, W_MIN], [A_MAX, W_MAX])


        # Second-Order CBF-QP Safety Filter

        p1 = state1[1:2]; psi1 = state1[3]; v1 = state1[4]
        p2 = state2[1:2]; psi2 = state2[3]; v2 = state2[4]

        delta_p = p1 - p2
        v1_vec = v1 * [cos(psi1), sin(psi1)]
        v2_vec = v2 * [cos(psi2), sin(psi2)]
        delta_v = v1_vec - v2_vec

        h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
        h_dot = 2 * dot(delta_p, delta_v)

        # Constraint: h_ddot + k1*h_dot + k2*h >= 0
        Lfh = 2*dot(delta_v, delta_v) + K1*h_dot + K2*h
        
        g_a1 = 2 * dot(delta_p, [cos(psi1), sin(psi1)])
        g_w1 = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])
        g_a2 = -2 * dot(delta_p, [cos(psi2), sin(psi2)])
        g_w2 = -2 * v2 * dot(delta_p, [-sin(psi2), cos(psi2)])
        
        A_cbf = [g_a1 g_w1 g_a2 g_w2] 
        b_cbf_lower = -Lfh

        # QP formulation: min (u-u_n)'*H*(u-u_n)
        H = diagm([
            20.0, 0.1,  # Costs for Vehicle 1 (a, w)
            20.0, 0.1   # Costs for Vehicle 2 (a, w)
        ])
        
        P = sparse(H * 2.0)
        q = -2.0 * H * [u_n1; u_n2]
        
        # THe Constraints
        A = sparse([A_cbf; I])
        l = [b_cbf_lower; A_MIN; W_MIN; A_MIN; W_MIN]
        u = [Inf; A_MAX; W_MAX; A_MAX; W_MAX]
        
        # Create and solve the OSQP problem (I think this is the right way to do it -- not sure though)
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        
        u_safe = [u_n1; u_n2] # Default to nominal if solver fails
        if results.info.status == :Solved
            u_safe = results.x
        else
            println("Warning: QP not solved at step $i. Using nominal control.")
        end

        u1_safe = u_safe[1:2]
        u2_safe = u_safe[3:4]

        # Update State
        state1 = state1 + vehicle_dynamics(state1, u1_safe) * DT
        state1[4] = clamp(state1[4], 0.0, V_MAX_1)
        
        state2 = state2 + vehicle_dynamics(state2, u2_safe) * DT 
        state2[4] = clamp(state2[4], 0.0, V_MAX_2)
        
        # Store History
        push!(hist_state1, state1)
        push!(hist_state2, state2)
        push!(hist_control1, u1_safe)
        push!(hist_control2, u2_safe)
    end

    println("Simulation finished.")
    return hist_state1, hist_state2, hist_control1, hist_control2
end


function plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2)
    println("Generating plots and animation...")
    
    # Extract data for plots
    num_steps_run = length(hist_state1) - 1
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)

    x1_hist = [s[1] for s in hist_state1]; y1_hist = [s[2] for s in hist_state1]
    v1_hist = [s[4] for s in hist_state1]; w1_hist = [c[2] for c in hist_control1[2:end]]
    
    x2_hist = [s[1] for s in hist_state2]; y2_hist = [s[2] for s in hist_state2]
    v2_hist = [s[4] for s in hist_state2]; w2_hist = [c[2] for c in hist_control2[2:end]]

    goal1_pos = [10.0, 1.0]
    theta = 0:0.1:(2*pi+0.1)

    # Generate Static Plots for Analysis
    p_traj = plot(x1_hist, y1_hist, label="V1 Path (Evader)", lw=2, aspect_ratio=:equal,
                  xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectories")
    plot!(p_traj, x2_hist, y2_hist, label="V2 Path (Pursuer)", lw=2)
    scatter!(p_traj, [goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)

    p_vel = plot(time_axis_states, v1_hist, label="V1 (Evader)", lw=2, title="Velocity Profiles")
    plot!(p_vel, time_axis_states, v2_hist, label="V2 (Pursuer)", lw=2, xlabel="Time [s]", ylabel="Velocity [m/s]")
    
    p_rot = plot(time_axis_controls, w1_hist, label="V1 (Evader)", lw=2, title="Rotational Speeds")
    plot!(p_rot, time_axis_controls, w2_hist, label="V2 (Pursuer)", lw=2, xlabel="Time [s]", ylabel="ω [rad/s]")

    static_plot = plot(p_traj, p_vel, p_rot, layout=(1,3), size=(1800, 500))
    
    plot_path = "cbf_analysis_plots_pursuit.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")


    # Generate Animation
    anim = @animate for i in 1:length(hist_state1)
        plot(x1_hist[1:i], y1_hist[1:i], label="V1 Path", lw=2, aspect_ratio=:equal,
             xlims=(-12, 12), ylims=(-6, 6),
             xlabel="x [m]", ylabel="y [m]", title="Pursuit-Evasion with CBF (Frame $i)")
        plot!(x2_hist[1:i], y2_hist[1:i], label="V2 Path", lw=2)

        # Plot Vehicle 1 (Evader)
        plot!(x1_hist[i] .+ R_AGENT .* cos.(theta), y1_hist[i] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 1", color=:blue)
        # Plot Vehicle 2 (Pursuer)
        plot!(x2_hist[i] .+ R_AGENT .* cos.(theta), y2_hist[i] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 2", color=:orange)
              
        # Plot the goal
        scatter!([goal1_pos[1]], [goal1_pos[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)
    end

    gif_path = "cbf_simulation_pursuit.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end


# Run Simulation and Generate GIF
hist_state1, hist_state2, hist_control1, hist_control2 = run_simulation()
plot_and_animate(hist_state1, hist_state2, hist_control1, hist_control2)

println("Script finished. Press Enter to exit...")
readline()

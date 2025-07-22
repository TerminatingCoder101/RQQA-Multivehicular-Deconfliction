# Required packages: OSQP for the QP solver, Plots for visualization,
# LinearAlgebra and SparseArrays for matrix operations.
using OSQP
using Plots
using LinearAlgebra
using SparseArrays

# --- Vehicle Definition ---
# Defines the state of the vehicle.
mutable struct Vehicle
    state::Vector{Float64} # State vector: [x position, y position, heading (psi), forward velocity (v)]
end

# --- Vehicle Dynamics ---
# Kinematic bicycle model.
function vehicle_dynamics(state, control)
    x, y, psi, v = state
    a, w = control # Control inputs: [acceleration, angular velocity]
    return [v * cos(psi), v * sin(psi), w, a]
end

# --- Simulation Parameters ---
const DT = 0.1          # Simulation time step [s]
const T_MAX = 20.0      # Maximum simulation time [s]
const N_STEPS = Int(T_MAX / DT) # Total number of simulation steps

# --- Safety and CBF Parameters ---
const R_AGENT = 0.5     # Radius of the agent [m]
const D_SAFE = 1.0      # Desired safety distance from obstacles [m]
const D_MIN_TOTAL = D_SAFE + R_AGENT # Total minimum distance (center-to-center)

# Gains for the second-order Control Barrier Function (CBF)
# These gains define how quickly the system responds to a potential collision.
# The constraint is h_ddot + K1*h_dot + K2*h >= 0
const K1 = 1.5
const K2 = 0.5

# --- Control and State Limits ---
const V_MIN, V_MAX = 0.0, 2.0   # Min/max velocity [m/s]
const A_MIN, A_MAX = -1.0, 1.0  # Min/max acceleration [m/s^2]
const W_MIN, W_MAX = -pi, pi    # Min/max angular velocity [rad/s]


"""
Main simulation function. It initializes the environment and runs the step-by-step simulation.
"""
function run_simulation()

    # --- Initialization ---
    agent = Vehicle([-10.0, 0.0, 0.0, V_MAX]) # Start agent on the left, moving towards the right
    goal = [10.0, 0.0]                       # Goal position on the right

    # History tracking for plotting
    hist_state = [copy(agent.state)]
    hist_control = [[0.0, 0.0]] # Initial control

    # Define two stationary obstacles
    obstacles = [
        Vehicle([-4.0, 0.0, 0.0, 0.0]),
        Vehicle([5.0, -2.5, 0.0, 0.0])
    ]
    num_obstacles = length(obstacles)

    println("Starting simulation with 1 Agent and $num_obstacles Obstacles...")

    # --- Simulation Loop ---
    for i in 1:N_STEPS
        # --- Check if goal is reached ---
        if norm(agent.state[1:2] - goal) < 0.3
            println("Goal reached at step $i. Stopping simulation.")
            break
        end

        # --- Nominal Controller ---
        # This controller defines the desired behavior (drive towards the goal)
        # without considering safety.
        error_vec = goal - agent.state[1:2]
        angle_to_goal = atan(error_vec[2], error_vec[1])
        psi_error = atan(sin(angle_to_goal - agent.state[3]), cos(angle_to_goal - agent.state[3]))

        # Simple proportional controller for velocity and heading
        a_n = 0.5 * (V_MAX - agent.state[4]) # Cruise control to reach V_MAX
        w_n = 2.0 * psi_error                # Proportional heading control
        u_n1 = clamp.([a_n, w_n], [A_MIN, W_MIN], [A_MAX, W_MAX])

        # --- Control Barrier Function (CBF) Constraints ---
        # Formulate linear constraints for the QP: A_cbf * u >= b_cbf
        # based on a second-order CBF: h_ddot + K1*h_dot + K2*h >= 0
        A_cbf = Matrix{Float64}(undef, num_obstacles, 2)
        b_cbf = Vector{Float64}(undef, num_obstacles)

        for (j, obs) in enumerate(obstacles)
            p1 = agent.state[1:2]; psi1 = agent.state[3]; v1 = agent.state[4]
            p2 = obs.state[1:2];   # Obstacle is stationary (v2=0)

            delta_p = p1 - p2
            v1_vec = v1 * [cos(psi1), sin(psi1)]
            delta_v = v1_vec # Since obstacle is stationary, delta_v = v1 - v2 = v1

            # Barrier function h and its derivatives
            h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
            h_dot = 2 * dot(delta_p, delta_v)

            # Decompose h_ddot into parts independent of control (Lfh_ddot) and dependent on control (Lgh_ddot * u)
            # h_ddot = Lfh_ddot + Lgh_ddot * u
            Lfh_ddot = 2 * dot(delta_v, delta_v)

            # Coefficients for control inputs u = [a, w]
            g_a = 2 * dot(delta_p, [cos(psi1), sin(psi1)])
            g_w = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])

            # The constraint is Lgh_ddot*u >= -(Lfh_ddot + K1*h_dot + K2*h)
            A_cbf[j, :] = [g_a g_w]
            b_cbf[j] = -(Lfh_ddot + K1 * h_dot + K2 * h)
        end

        # --- QP Solver (OSQP) ---
        # We find a safe control u = [a, w] that is as close as possible
        # to the nominal control u_n1, while satisfying the CBF and control limit constraints.
        # Objective: minimize ||u - u_n1||^2_H
        # This is equivalent to: 0.5*u'*P*u + q'*u

        # Weighting matrix for the cost function.
        # A higher penalty on acceleration encourages smoother speed changes.
        H_cost = diagm([
            1.0,  # Penalty on acceleration 'a'
            0.1   # Penalty on angular velocity 'w'
        ])
        P = sparse(2.0 * H_cost)
        q = -2.0 * H_cost * u_n1

        # Assemble the full constraint matrix A and bound vectors l, u for OSQP
        # The constraints are:
        # 1. CBF constraints: A_cbf * u >= b_cbf
        # 2. Box constraints: [A_MIN, W_MIN] <= u <= [A_MAX, W_MAX]
        # We write this in the standard form l <= A_qp * u <= u_qp
        A_qp = sparse([A_cbf; I(2)])
        l_qp = [b_cbf; A_MIN; W_MIN]
        u_qp = [fill(Inf, num_obstacles); A_MAX; W_MAX]

        # Setup and solve the QP
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A_qp, l=l_qp, u=u_qp, verbose=false, eps_abs=1e-4, eps_rel=1e-4)
        results = OSQP.solve!(model)

        u_safe = u_n1 # Fallback to nominal control if solver fails
        if results.info.status == :Solved
            u_safe = results.x
        else
            println("Warning: QP not solved at step $i (status: $(results.info.status)). Using nominal control.")
        end

        # --- Update State ---
        agent.state .+= vehicle_dynamics(agent.state, u_safe) * DT
        agent.state[4] = clamp(agent.state[4], V_MIN, V_MAX) # Enforce velocity limits

        push!(hist_state, copy(agent.state))
        push!(hist_control, u_safe)
    end

    println("Simulation finished.")
    return hist_state, hist_control, obstacles, goal
end


"""
Generates and saves plots and an animation of the simulation results.
"""
function plot_and_animate(hist_state, hist_control, obstacles, goal)
    println("Generating plots and animation...")

    num_steps_run = length(hist_state) - 1
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)

    # Extract data for plotting
    x_hist = [s[1] for s in hist_state]
    y_hist = [s[2] for s in hist_state]
    v_hist = [s[4] for s in hist_state]
    # The first control is a placeholder, so we skip it
    w_hist = [c[2] for c in hist_control[2:end]]
    a_hist = [c[1] for c in hist_control[2:end]]

    # Circle points for plotting
    theta = 0:0.1:(2*pi+0.1)

    # --- Generate Static Plots for Analysis ---
    p_traj = plot(x_hist, y_hist, label="Agent Path", lw=2, aspect_ratio=:equal,
                  xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectory")
    for (i, obs) in enumerate(obstacles)
        obs_pos = obs.state[1:2]
        scatter!(p_traj, [obs_pos[1]], [obs_pos[2]], label="Obs $i", marker=:star5, markersize=8, color=:black)
        # Plot the safety boundary around the obstacle
        plot!(p_traj, obs_pos[1] .+ D_MIN_TOTAL .* cos.(theta), obs_pos[2] .+ D_MIN_TOTAL .* sin.(theta),
              seriestype=:shape, fillalpha=0.2, lw=0, label="", color=:red)
    end
    scatter!(p_traj, [goal[1]], [goal[2]], label="Goal", marker=:xcross, markersize=8, color=:green)

    p_vel = plot(time_axis_states, v_hist, label="Agent Velocity", lw=2,
                 xlabel="Time [s]", ylabel="Velocity [m/s]", title="Velocity Profile", legend=:bottomright)

    p_ctrl = plot(time_axis_controls, [a_hist w_hist], label=["Acceleration" "Angular Velocity"], lw=2,
                  xlabel="Time [s]", ylabel="Control Input", title="Control History", legend=:bottomright)

    static_plot = plot(p_traj, p_vel, p_ctrl, layout=(1,3), size=(1800, 500))

    plot_path = "cbf_mvso.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")


    # --- Generate Animation ---
    anim = @animate for i in 1:length(hist_state)
        # Create the base plot for each frame
        p_anim = plot(xlims=(-12, 12), ylims=(-8, 8), aspect_ratio=:equal,
                      xlabel="x [m]", ylabel="y [m]", title="CBF Obstacle Avoidance (Step $i)")

        # Plot current agent path
        plot!(p_anim, x_hist[1:i], y_hist[1:i], label="Agent Path", lw=2, color=:blue)

        # Plot agent body
        plot!(p_anim, x_hist[i] .+ R_AGENT .* cos.(theta), y_hist[i] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Agent", color=:blue)

        # Plot obstacles and their safety boundaries
        for (k, obs) in enumerate(obstacles)
            obs_pos = obs.state[1:2]
            scatter!(p_anim, [obs_pos[1]], [obs_pos[2]], label=(k==1 ? "Obstacle" : ""), marker=:star5, markersize=8, color=:black)
            plot!(p_anim, obs_pos[1] .+ D_MIN_TOTAL .* cos.(theta), obs_pos[2] .+ D_MIN_TOTAL .* sin.(theta),
                  seriestype=:shape, fillalpha=0.2, lw=0, label=(k==1 ? "Safety Zone" : ""), color=:red)
        end

        # Plot the goal
        scatter!(p_anim, [goal[1]], [goal[2]], label="Goal", marker=:xcross, markersize=8, color=:green, legend=:topleft)
    end

    gif_path = "cbf_mvso.gif"
    gif(anim, gif_path, fps = 20)
    println("Animation saved to $gif_path")
end

hist_state, hist_control, obstacles, goal = run_simulation()
plot_and_animate(hist_state, hist_control, obstacles, goal)

println("Script finished. Press Enter to exit...")
readline()

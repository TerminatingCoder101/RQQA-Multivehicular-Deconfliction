using OSQP
using Plots
using LinearAlgebra
using SparseArrays

mutable struct Vehicle
    state::Vector{Float64} # [x, y, psi, v]
end

function vehicle_dynamics(state, control)
    x, y, psi, v = state
    a, w = control
    return [v * cos(psi), v * sin(psi), w, a]
end

const DT = 0.1      # Time step [s]
const T_MAX = 20.0    # Max simulation time [s]
const N_STEPS = Int(T_MAX / DT)


const R_AGENT = 0.5 
const D_SAFE = 1.0 
const D_MIN_TOTAL = D_SAFE + R_AGENT

const K1 = 1.5
const K2 = 0.5

# Control limits
const V_MIN, V_MAX = 0.0, 2.0
const A_MIN, A_MAX = -1.0, 1.0
const W_MIN, W_MAX = -pi, pi


function run_simulation()

    agent = Vehicle([-10.0, 0.0, 0.0, V_MAX])
    goal = [10.0, 0.0]
    hist_state = [agent.state]
    hist_control = [[0.0, 0.0]]

    obstacles = [
        Vehicle([-3.0, 4.0, 0.0, 0.0]),
        Vehicle([3.0, -4.0, 0.0, 0.0])
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
        error1 = goal - agent.state[1:2]
        angle_to_goal1 = atan(error1[2], error1[1])
        psi_error1 = atan(sin(angle_to_goal1 - agent.state[3]), cos(angle_to_goal1 - agent.state[3]))
        a_n1 = 0.5 * (V_MAX - agent.state[4])
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([a_n1, w_n1], [A_MIN, W_MIN], [A_MAX, W_MAX])
        
        A_cbf = Matrix{Float64}(undef, num_obstacles, 2)
        b_cbf_lower = Vector{Float64}(undef, num_obstacles)

        for (j, obs) in enumerate(obstacles)
            p1 = agent.state[1:2]; psi1 = agent.state[3]; v1 = agent.state[4]
            p2 = obs.state[1:2]
            
            delta_p = p1 - p2
            v1_vec = v1 * [cos(psi1), sin(psi1)]
            delta_v = v1_vec 

            h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
            h_dot = 2 * dot(delta_p, delta_v)
            
            Lfh_j = 2*dot(delta_v, delta_v) + K1*h_dot + K2*h
            
            g_a1 = 2 * dot(delta_p, [cos(psi1), sin(psi1)])
            g_w1 = 2 * v1 * dot(delta_p, [-sin(psi1), cos(psi1)])
            
            A_cbf[j, :] = [g_a1 g_w1]
            b_cbf_lower[j] = -Lfh_j
        end

        # QP formulation for the agent's control u = [a, w]
        H = diagm([10.0, 0.1])
        P = sparse(H * 2.0)
        q = -2.0 * H * u_n1
        
        A = sparse([A_cbf; I])
        l = [b_cbf_lower; A_MIN; W_MIN]
        u = [fill(Inf, num_obstacles); A_MAX; W_MAX]
        
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        
        u_safe = u_n1
        if results.info.status == :Solved
            u_safe = results.x
        else
            println("Warning: QP not solved at step $i. Using nominal control.")
        end

        # --- Update State ---
        agent.state .+= vehicle_dynamics(agent.state, u_safe) * DT
        agent.state[4] = clamp(agent.state[4], V_MIN, V_MAX)
        
        push!(hist_state, agent.state)
        push!(hist_control, u_safe)
    end

    println("Simulation finished.")
    return hist_state, hist_control, obstacles, goal
end


function plot_and_animate(hist_state, hist_control, obstacles, goal)
    println("Generating plots and animation...")
    
    num_steps_run = length(hist_state) - 1
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)

    x1_hist = [s[1] for s in hist_state]
    y1_hist = [s[2] for s in hist_state]
    v1_hist = [s[4] for s in hist_state]
    w1_hist = [c[2] for c in hist_control[2:end]]
    
    theta = 0:0.1:(2*pi+0.1)

    # --- Generate Static Plots for Analysis ---
    p_traj = plot(x1_hist, y1_hist, label="Agent Path", lw=2, aspect_ratio=:equal,
                  xlabel="x [m]", ylabel="y [m]", title="Vehicle Trajectory")
    for (i, obs) in enumerate(obstacles)
        obs_pos = obs.state[1:2]
        scatter!(p_traj, [obs_pos[1]], [obs_pos[2]], label="Obs $i", marker=:star5, markersize=8, color=:black)
        plot!(p_traj, obs_pos[1] .+ D_MIN_TOTAL .* cos.(theta), obs_pos[2] .+ D_MIN_TOTAL .* sin.(theta),
              seriestype=:shape, fillalpha=0.2, lw=0, label="", color=:red)
    end
    scatter!(p_traj, [goal[1]], [goal[2]], label="Goal", marker=:xcross, markersize=8, color=:green)

    p_vel = plot(time_axis_states, v1_hist, label="Agent Velocity", lw=2,
                 xlabel="Time [s]", ylabel="Velocity [m/s]", title="Velocity Profile", legend=:bottomright)
    
    p_rot = plot(time_axis_controls, w1_hist, label="Agent ω", lw=2,
                 xlabel="Time [s]", ylabel="ω [rad/s]", title="Rotational Speed", legend=:bottomright)

    static_plot = plot(p_traj, p_vel, p_rot, layout=(1,3), size=(1800, 500))
    
    plot_path = "cbf_analysis_plots_two_obstacles.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")


    # --- Generate Animation ---
    anim = @animate for i in 1:length(hist_state)
        # Create the base plot for each frame
        p_anim = plot(xlims=(-12, 12), ylims=(-8, 8), aspect_ratio=:equal,
                      xlabel="x [m]", ylabel="y [m]", title="Multi-Obstacle Avoidance (Frame $i)")

        plot!(p_anim, x1_hist, y1_hist, label="", lw=1, color=:lightgray)
        
        plot!(p_anim, x1_hist[1:i], y1_hist[1:i], label="Agent Path", lw=2, color=:blue)

        plot!(p_anim, x1_hist[i] .+ R_AGENT .* cos.(theta), y1_hist[i] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Agent", color=:blue)

        for obs in obstacles
            obs_pos = obs.state[1:2]
            scatter!(p_anim, [obs_pos[1]], [obs_pos[2]], label="", marker=:star5, markersize=8, color=:black)
            plot!(p_anim, obs_pos[1] .+ D_MIN_TOTAL .* cos.(theta), obs_pos[2] .+ D_MIN_TOTAL .* sin.(theta),
                  seriestype=:shape, fillalpha=0.2, lw=0, label="", color=:red)
        end
              
        # Plot the goal
        scatter!(p_anim, [goal[1]], [goal[2]], label="Goal", marker=:xcross, markersize=8, color=:green)
    end

    gif_path = "cbf_simulation_two_obstacles.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end


hist_state, hist_control, obstacles, goal = run_simulation()
plot_and_animate(hist_state, hist_control, obstacles, goal)

println("Script finished. Press Enter to exit...")
readline()

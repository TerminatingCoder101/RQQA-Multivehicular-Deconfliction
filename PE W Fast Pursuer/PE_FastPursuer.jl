
using OSQP
using Plots
using LinearAlgebra
using SparseArrays


mutable struct Vehicle
    state::Vector{Float64} # [x, y, psi, v]
    control::Vector{Float64} # [a, w]
    hist_state::Vector{Vector{Float64}}
    hist_control::Vector{Vector{Float64}}
    v_max::Float64
end


# Vehicle Dynamics Model: Second-order unicycle model
function vehicle_dynamics(state, control)
    x, y, psi, v = state
    a, w = control

    x_dot = v * cos(psi)
    y_dot = v * sin(psi)
    psi_dot = w
    v_dot = a

    return [x_dot, y_dot, psi_dot, v_dot]
end

# --- Proportional Navigation Logic ---
# Calculates the heading for the pursuer to intercept the evader
function calculate_intercept_heading(pursuer, evader)
    P = pursuer.state[1:2]
    E = evader.state[1:2]
    Vp = pursuer.v_max 
    Ve = evader.state[4]


    if Ve < 1e-3
        return atan(E[2] - P[2], E[1] - P[1])
    end

    psi_e = evader.state[3]
    Ve_vec = Ve * [cos(psi_e), sin(psi_e)]

    # Vector from pursuer to evader
    PE = E - P
    dist_PE = norm(PE)

    # Angle between the vector from Evader to Pursuer (EP) and the Evader's velocity vector
    cos_alpha = dot(-PE, Ve_vec) / (dist_PE * Ve + 1e-9)

    # Coefficients for the quadratic equation: a*t^2 + b*t + c = 0
    # where t is the time to intercept.
    # This is derived from the Law of Cosines on the velocity triangle.
    a = Vp^2 - Ve^2
    b = 2 * dist_PE * Ve * cos_alpha
    c = -dist_PE^2

    # Solve the quadratic equation for time 't'
    discriminant = b^2 - 4*a*c
    
    t_intercept = -1.0
    if discriminant >= 0
        # Two potential solutions for time
        t1 = (-b + sqrt(discriminant)) / (2*a + 1e-9)
        t2 = (-b - sqrt(discriminant)) / (2*a + 1e-9)
        
        # We need the smallest positive real time to intercept
        if t1 > 0 && t2 > 0
            t_intercept = min(t1, t2)
        elseif t1 > 0
            t_intercept = t1
        elseif t2 > 0
            t_intercept = t2
        end
    end

    # If a valid intercept time is found, calculate the intercept point
    if t_intercept > 0
        intercept_point = E + Ve_vec * t_intercept
        # Return the heading towards the intercept point
        return atan(intercept_point[2] - P[2], intercept_point[1] - P[1])
    else
        # Fallback: if no intercept is possible, just aim at the evader's current position
        return atan(E[2] - P[2], E[1] - P[1])
    end
end


# Simulation parameters
const DT = 0.1 # Time step [s]
const T_MAX = 30.0
const N_STEPS = Int(T_MAX / DT)

# Radius of Agent
const R_AGENT = 0.5

# Min Safe distance
const D_SAFE = 1.0
const D_MIN_TOTAL = 2 * R_AGENT + D_SAFE 

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 2.0
const K2 = 1.0

# Control limits
const V_MAX_1 = 2.0 
const V_MAX_2 = 3.0
const A_MIN, A_MAX = -1.0, 1.0
const A_LAT_MAX = 4.0 # Max lateral acceleration for Pachter model constraint

# --- 2. Main Simulation Function ---

function run_simulation()

    # Vehicle 1 (Evader): Starts on the left, goal is on the right
    evader = Vehicle(
        [-10.0, 1.0, 0.0, V_MAX_1],
        [0.0, 0.0],
        [[-10.0, 1.0, 0.0, V_MAX_1]],
        [[0.0, 0.0]],
        V_MAX_1
    )
    goal1 = [10.0, 1.0]

    # Vehicle 2 (Pursuer): Starts on the right, goal is Vehicle 1
    pursuer = Vehicle(
        [5.0, -1.0, pi, V_MAX_2],
        [0.0, 0.0],
        [[5.0, -1.0, pi, V_MAX_2]],
        [[0.0, 0.0]],
        V_MAX_2
    )

    # History for debugging plots
    hist_h = Float64[]
    hist_h_dot = Float64[]
    hist_w_bounds1 = Float64[]
    hist_w_bounds2 = Float64[]


    println("Starting simulation...")

    # Simulation Loop
    for i in 1:N_STEPS
        # Check if the goal has been reached or not
        if norm(evader.state[1:2] - goal1) < 0.3
            println("Evader reached goal at step $i. Stopping simulation.")
            # Trim history to actual run length
            hist_h = hist_h[1:i-1]
            hist_h_dot = hist_h_dot[1:i-1]
            hist_w_bounds1 = hist_w_bounds1[1:i-1]
            hist_w_bounds2 = hist_w_bounds2[1:i-1]
            break
        end

        # --- Calculate Dynamic Angular Velocity Bounds (Pachter Model) ---
        w_max_1 = A_LAT_MAX / (evader.state[4] + 1e-6)
        w_max_2 = A_LAT_MAX / (pursuer.state[4] + 1e-6)
        push!(hist_w_bounds1, w_max_1)
        push!(hist_w_bounds2, w_max_2)


        # Nominal Controller for Vehicle 1 (Evader)
        error1 = goal1 - evader.state[1:2]
        angle_to_goal1 = atan(error1[2], error1[1])
        psi_error1 = atan(sin(angle_to_goal1 - evader.state[3]), cos(angle_to_goal1 - evader.state[3]))
        a_n1 = 0.5 * (evader.v_max - evader.state[4])
        w_n1 = 2.0 * psi_error1
        u_n1 = clamp.([a_n1, w_n1], [A_MIN, -w_max_1], [A_MAX, w_max_1])

        # Nominal Controller for Vehicle 2 (Pursuer) using Proportional Navigation
        angle_to_intercept = calculate_intercept_heading(pursuer, evader)
        psi_error2 = atan(sin(angle_to_intercept - pursuer.state[3]), cos(angle_to_intercept - pursuer.state[3]))
        a_n2 = 0.5 * (pursuer.v_max - pursuer.state[4])
        w_n2 = 2.0 * psi_error2
        u_n2 = clamp.([a_n2, w_n2], [A_MIN, -w_max_2], [A_MAX, w_max_2])


        # Second-Order CBF-QP Safety Filter
        p1 = evader.state[1:2]; psi1 = evader.state[3]; v1 = evader.state[4]
        p2 = pursuer.state[1:2]; psi2 = pursuer.state[3]; v2 = pursuer.state[4]
        
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
        
        A = sparse([A_cbf; I])
        l = [b_cbf_lower; A_MIN; -w_max_1; A_MIN; -w_max_2]
        u = [Inf; A_MAX; w_max_1; A_MAX; w_max_2]
        
        model = OSQP.Model()
        OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
        results = OSQP.solve!(model)
        
        u_safe = [u_n1; u_n2]
        if results.info.status == :Solved
            u_safe = results.x
        else
            println("Warning: QP not solved at step $i. Using nominal control.")
        end

        u1_safe = u_safe[1:2]
        u2_safe = u_safe[3:4]

        # Update State
        evader.state = evader.state + vehicle_dynamics(evader.state, u1_safe) * DT
        evader.state[4] = clamp(evader.state[4], 0.0, V_MAX_1)
        
        pursuer.state = pursuer.state + vehicle_dynamics(pursuer.state, u2_safe) * DT 
        pursuer.state[4] = clamp(pursuer.state[4], 0.0, V_MAX_2)
        
        # Store History
        push!(evader.hist_state, evader.state)
        push!(pursuer.hist_state, pursuer.state)
        push!(evader.hist_control, u1_safe)
        push!(pursuer.hist_control, u2_safe)
    end

    println("Simulation finished.")
    return evader, pursuer, goal1, hist_h, hist_h_dot, hist_w_bounds1, hist_w_bounds2
end


function plot_and_animate(evader, pursuer, goal, hist_h, hist_h_dot, hist_w_bounds1, hist_w_bounds2)
    println("Generating plots and animation...")
    
    # --- Extract data for plots ---
    num_steps_run = length(hist_h)
    time_axis_states = 0:DT:(num_steps_run * DT)
    time_axis_controls = 0:DT:((num_steps_run-1) * DT)

    # --- Generate Comprehensive Static Plots for Analysis ---
    plot_title = "Pursuit-Evasion Analysis (D_SAFE=$(D_SAFE), K1=$(K1), K2=$(K2), A_LAT_MAX=$(A_LAT_MAX))"

    # 1. Trajectory Plot
    p_traj = plot(
        [s[1] for s in evader.hist_state], [s[2] for s in evader.hist_state],
        label="Evader Path", lw=2, aspect_ratio=:equal,
        xlabel="Position X [DU]", ylabel="Position Y [DU]", title="Vehicle Trajectories"
    )
    plot!(p_traj, [s[1] for s in pursuer.hist_state], [s[2] for s in pursuer.hist_state], label="Pursuer Path", lw=2)
    scatter!(p_traj, [goal[1]], [goal[2]], label="Evader Goal", marker=:xcross, markersize=8, color=:green)

    # 2. State Variables Plot (Velocity)
    p_vel = plot(time_axis_states, [s[4] for s in evader.hist_state], label="Evader", lw=2, title="State: Velocity", xlabel="Time [s]", ylabel="Velocity [m/s]")
    plot!(p_vel, time_axis_states, [s[4] for s in pursuer.hist_state], label="Pursuer", lw=2)
    hline!(p_vel, [V_MAX_1, V_MAX_2], linestyle=:dash, label=["V1_max", "V2_max"])

    # 3. Control Inputs Plot
    p_accel = plot(time_axis_controls, [c[1] for c in evader.hist_control[2:end]], label="Evader", lw=2, title="Control: Acceleration", xlabel="Time [s]", ylabel="a [m/s^2]")
    plot!(p_accel, time_axis_controls, [c[1] for c in pursuer.hist_control[2:end]], label="Pursuer", lw=2)
    hline!(p_accel, [A_MAX, A_MIN], linestyle=:dash, color=:red, label="Bounds")
    
    p_rot = plot(time_axis_controls, [c[2] for c in evader.hist_control[2:end]], label="Evader", lw=2, title="Control: Angular Velocity", xlabel="Time [s]", ylabel="ω [rad/s]")
    plot!(p_rot, time_axis_controls, [c[2] for c in pursuer.hist_control[2:end]], label="Pursuer", lw=2)
    plot!(p_rot, time_axis_controls, hist_w_bounds1, linestyle=:dash, color=:red, label="V1 Bounds")
    plot!(p_rot, time_axis_controls, -hist_w_bounds1, linestyle=:dash, color=:red, label="")
    plot!(p_rot, time_axis_controls, hist_w_bounds2, linestyle=:dash, color=:purple, label="V2 Bounds")
    plot!(p_rot, time_axis_controls, -hist_w_bounds2, linestyle=:dash, color=:purple, label="")
    p_ctrl = plot(p_accel, p_rot, layout=(2,1))

    # 4. CBF Values Plot
    p_h = plot(time_axis_controls, hist_h, label="h(x)", lw=2, title="CBF Value (Safety Metric)", xlabel="Time [s]", ylabel="h(x) [DU^2]")
    hline!(p_h, [0], linestyle=:dash, color=:red, label="Boundary (h=0)")
    
    p_h_dot = plot(time_axis_controls, hist_h_dot, label="h_dot(x)", lw=2, title="CBF Derivative (Closure Rate)", xlabel="Time [s]", ylabel="h_dot(x) [DU^2/s]")
    p_cbf = plot(p_h, p_h_dot, layout=(2,1))

    # Combine all plots into a 2x2 grid
    static_plot = plot(p_traj, p_vel, p_ctrl, p_cbf, layout=(2,2), size=(1200, 1000), plot_title=plot_title)
    
    plot_path = "cbf_analysis_plots_pursuit.png"
    savefig(static_plot, plot_path)
    println("Analysis plots saved to $plot_path")


    # --- Generate Animation ---
    anim = @animate for i in 1:length(evader.hist_state)
        plot([s[1] for s in evader.hist_state[1:i]], [s[2] for s in evader.hist_state[1:i]], label="V1 Path", lw=2, aspect_ratio=:equal,
             xlims=(-14, 14), ylims=(-8, 8),
             xlabel="x [DU]", ylabel="y [DU]", title="Pursuit-Evasion with CBF (Frame $i)")
        plot!([s[1] for s in pursuer.hist_state[1:i]], [s[2] for s in pursuer.hist_state[1:i]], label="V2 Path", lw=2)

        theta = 0:0.1:(2*pi+0.1)
        # Plot Vehicle 1 (Evader)
        plot!([evader.hist_state[i][1]] .+ R_AGENT .* cos.(theta), [evader.hist_state[i][2]] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 1", color=:blue)
        # Plot Vehicle 2 (Pursuer)
        plot!([pursuer.hist_state[i][1]] .+ R_AGENT .* cos.(theta), [pursuer.hist_state[i][2]] .+ R_AGENT .* sin.(theta),
              seriestype=:shape, fillalpha=0.3, lw=0, label="Vehicle 2", color=:orange)
              
        # Plot the goal
        scatter!([goal[1]], [goal[2]], label="V1 Goal", marker=:xcross, markersize=8, color=:green)
    end

    gif_path = "cbf_simulation_pursuit.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end


# --- Run Simulation and Generate GIF ---
evader, pursuer, goal, hist_h, hist_h_dot, hist_w_bounds1, hist_w_bounds2 = run_simulation()
plot_and_animate(evader, pursuer, goal, hist_h, hist_h_dot, hist_w_bounds1, hist_w_bounds2)

println("Script finished. Press Enter to exit...")
readline()
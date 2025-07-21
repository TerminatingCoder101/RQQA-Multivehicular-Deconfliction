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
const DT = 0.1 
const T_MAX = 30.0
const N_STEPS = Int(T_MAX / DT)

# Agent Properties
const R_AGENT = 0.5
const D_MIN_TOTAL = 3 * R_AGENT + 1 
const D_THREAT = 5.0

# Second-order CBF parameters (h_ddot + K1*h_dot + K2*h >= 0)
const K1 = 3.0
const K2 = 2.0

# Control limits
const V_MAX_E = 2.0
const V_MAX_P = 3.0
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
    interceptionHist = [[], [], []]
    state1 = [-10.0, 2.0, 0.0, 0.0]; goal1 = [10.0, 2.0]  # Evader 1
    state2 = [-10.0, -2.0, 0.0, 0.0]; goal2 = [10.0, -2.0] # Evader 2
    state3 = [10.0, 4.0, pi, V_MAX_P]   # Pursuer 1
    state4 = [10.0, 0.0, pi, V_MAX_P] # Pursuer 2
    state5 = [10.0, -4.0, pi, V_MAX_P] # Pursuer 3

    states = [state1, state2, state3, state4, state5]
    hist_states = [[s] for s in states]
    hist_controls = [[[0.0, 0.0]] for _ in states]    

    println("Starting 2 vs 3...")

    for i in 1:N_STEPS
        # Check for goal completion
        if norm(states[1][1:2] - goal1) < 0.5 && norm(states[2][1:2] - goal2) < 0.5
            println("Both evaders reached goal at step $i. Stopping simulation.")
            break
        end

        # --- Nominal Controllers ---
        
        # Evader Controllers (E1, E2)
        u_nominal = Vector{Vector{Float64}}(undef, 5)
        for evader_idx in 1:2
            current_state = states[evader_idx]
            goal = (evader_idx == 1) ? goal1 : goal2
            
            error_vec = goal - current_state[1:2]
            dist_to_goal = norm(error_vec)
            
            # Find closest pursuer
            closest_pursuer_dist = Inf
            closest_pursuer_state = states[3]
            for pursuer_idx in 3:5
                dist = norm(current_state[1:2] - states[pursuer_idx][1:2])
                if dist < closest_pursuer_dist
                    closest_pursuer_dist = dist
                    closest_pursuer_state = states[pursuer_idx]
                end
            end

            angle_to_goal = atan(error_vec[2], error_vec[1]) # Determines straight line angel to goal
            final_angle_command = angle_to_goal # Create new final angle command (that could be changed)

            if closest_pursuer_dist < D_THREAT
                vec_to_goal = normalize(error_vec) # Unit vector of goal vector 
                vec_away_from_pursuer = normalize(current_state[1:2] - closest_pursuer_state[1:2]) # Unit vector away from pursuer
                evasive_weight = 1.0 - (closest_pursuer_dist / D_THREAT) # Create gradient of safety 
                # Barebone function: 1.0 * vec_to_goal + 0.0 * vec_away_from_pursuer for pursuer far (go to goal)
                # If pursuer is close then evasive_weight increases making vec_away_from_pursuer larger.
                desired_direction_vec = (1.0 - evasive_weight) * vec_to_goal + evasive_weight * vec_away_from_pursuer 
                final_angle_command = atan(desired_direction_vec[2], desired_direction_vec[1])
            end
            
            psi_error = atan(sin(final_angle_command - current_state[3]), cos(final_angle_command - current_state[3]))
            
            v_goal_based = 0.9 * dist_to_goal
            v_turn_based = V_MAX_E * (1.0 - 0.5 * abs(psi_error) / pi)
            v_evasive = V_MAX_E
            
            v_desired = min(v_goal_based, v_turn_based, v_evasive)
            a_n = 0.5 * (v_desired - current_state[4])
            w_n = 2.0 * psi_error
            u_nominal[evader_idx] = clamp.([a_n, w_n], [A_MIN, W_MIN], [A_MAX, W_MAX])
        end

        # Pursuer Controllers (P1, P2, P3)
        for pursuer_idx in 3:5
            current_state = states[pursuer_idx]
            
            dist_to_e1 = norm(current_state[1:2] - states[1][1:2])
            dist_to_e2 = norm(current_state[1:2] - states[2][1:2])
            target_evader_state = (dist_to_e1 < dist_to_e2) ? states[1] : states[2]

            intercept_pt = calc_intercept_pt(target_evader_state, current_state, V_MAX_P)
            push!(interceptionHist[pursuer_idx-2], intercept_pt)
            
            error_vec = intercept_pt - current_state[1:2]
            angle_to_goal = atan(error_vec[2], error_vec[1])
            psi_error = atan(sin(angle_to_goal - current_state[3]), cos(angle_to_goal - current_state[3]))
            
            a_n = abs(psi_error) > pi / 4.0 ? A_MIN : 0.5 * (V_MAX_P - current_state[4])
            w_n = 2.0 * psi_error
            u_nominal[pursuer_idx] = clamp.([a_n, w_n], [A_MIN, W_MIN], [A_MAX, W_MAX])
        end

        # --- CBF Safety Filter for Evaders ---
        u_safe = copy(u_nominal)
        
        for evader_idx in 1:2
            function get_cbf_terms(evader_state, other_state, other_control)
                p_e = evader_state[1:2]; psi_e = evader_state[3]; v_e = evader_state[4]
                p_o = other_state[1:2]; psi_o = other_state[3]; v_o = other_state[4]
                a_o, w_o = other_control
                delta_p = p_e - p_o
                v_e_vec = v_e * [cos(psi_e), sin(psi_e)]
                v_o_vec = v_o * [cos(psi_o), sin(psi_o)]
                delta_v = v_e_vec - v_o_vec

                h = dot(delta_p, delta_p) - D_MIN_TOTAL^2
                h_dot = 2 * dot(delta_p, delta_v)
                a_o_vec = a_o * [cos(psi_o), sin(psi_o)] - v_o * w_o * [sin(psi_o), -cos(psi_o)]
                h_ddot_part = 2*dot(delta_v, delta_v) + 2*dot(delta_p, -a_o_vec) + K1*h_dot + K2*h
                g_ae = 2 * dot(delta_p, [cos(psi_e), sin(psi_e)])
                g_we = 2 * v_e * dot(delta_p, [-sin(psi_e), cos(psi_e)])
                return [g_ae g_we], -h_ddot_part, h
            end

            A_cbf_list = []
            b_cbf_list = []
            other_indices = filter(x -> x != evader_idx, 1:5)

            for other_idx in other_indices
                A_row, b_val, h_val = get_cbf_terms(states[evader_idx], states[other_idx], u_nominal[other_idx])
                if h_val < D_MIN_TOTAL^2 * 1.5
                    push!(A_cbf_list, A_row)
                    push!(b_cbf_list, b_val)
                end
            end

            if !isempty(A_cbf_list)
                A_cbf = vcat(A_cbf_list...)
                b_cbf_lower = vcat(b_cbf_list...)
                H = diagm([20.0, 0.1])
                P = sparse(H * 2.0)
                q = -2.0 * H * u_nominal[evader_idx]
                A = sparse([A_cbf; I(2)])
                l = [b_cbf_lower; A_MIN; W_MIN]
                u = [fill(Inf, size(A_cbf, 1)); A_MAX; W_MAX]
                model = OSQP.Model()
                OSQP.setup!(model; P=P, q=q, A=A, l=l, u=u, verbose=false, eps_abs=1e-5, eps_rel=1e-5)
                results = OSQP.solve!(model)
                if results.info.status == :Solved
                    u_safe[evader_idx] = results.x
                end
            else
                println("List is empty")
            end
        end

        # --- Update States ---
        for i in 1:5
            states[i] = states[i] + vehicle_dynamics(states[i], u_safe[i]) * DT
            max_v = (i <= 2) ? V_MAX_E : V_MAX_P
            states[i][4] = clamp(states[i][4], 0.0, max_v)
            push!(hist_states[i], copy(states[i]))
            push!(hist_controls[i], u_safe[i])
        end
    end

    println("Simulation finished.")
    return hist_states, hist_controls, interceptionHist
end


function plot_and_animate(hist_states, hist_controls, interceptionHist)
    println("Generating plots and animation...")
    
    # Extract histories
    x_hists = [[s[1] for s in hist] for hist in hist_states]
    y_hists = [[s[2] for s in hist] for hist in hist_states]
    
    goal1_pos = [10.0, 2.0]
    goal2_pos = [10.0, -2.0]
    theta = 0:0.1:(2*pi+0.1)
    
    colors = [:blue, :cyan, :red, :orange, :purple]
    labels = ["E1", "E2", "P1", "P2", "P3"]

    anim = @animate for i in 1:lastindex(interceptionHist[1])
        p_anim = plot(aspect_ratio=:equal, xlims=(-12, 12), ylims=(-8, 8),
             xlabel="x [m]", ylabel="y [m]", title="2v3 Pursuit-Evasion (Frame $i)")
        
        # Plot paths and current positions
        for j in 1:5
            plot!(p_anim, x_hists[j][1:i], y_hists[j][1:i], label="", lw=2, c=colors[j])
            plot!(p_anim, x_hists[j][i].+R_AGENT.*cos.(theta), y_hists[j][i].+R_AGENT.*sin.(theta), 
                  seriestype=:shape, fillalpha=0.4, lw=0, label=labels[j], c=colors[j])
        end
        
        # Plot goals and intercept points
        scatter!(p_anim, [goal1_pos[1]], [goal1_pos[2]], label="G1", marker=:xcross, markersize=8, color=:blue)
        scatter!(p_anim, [goal2_pos[1]], [goal2_pos[2]], label="G2", marker=:xcross, markersize=8, color=:cyan)
        
        for p_idx in 1:3
             scatter!(p_anim, [interceptionHist[p_idx][i][1]], [interceptionHist[p_idx][i][2]], 
                      label="IP$(p_idx)", marker=:star5, markersize=6, color=colors[p_idx+2])
        end
    end

    gif_path = "cbf_simulation_2v3.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end

# Run Simulation and Generate GIF
hist_states, hist_controls, interceptionHist = run_simulation()
plot_and_animate(hist_states, hist_controls, interceptionHist)

println("Script finished. Press Enter to exit...")
readline()

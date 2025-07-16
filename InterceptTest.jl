using Plots
using LinearAlgebra

#==============================================================================
# INTERCEPT LOGIC TEST SCRIPT
# This script tests the core intercept calculation in isolation.
# - No complex controllers are used.
# - Both vehicles move at a constant velocity in a straight line.
# - The pursuer is faster than the evader.
# The goal is to verify that the vehicles meet at the predicted intercept point.
==============================================================================#


# --- Simulation Parameters ---
const DT = 0.1 
const T_MAX = 20.0
const N_STEPS = Int(T_MAX / DT)

# --- Vehicle Properties ---
const V_EVADER = 1.5  # Constant speed for the evader
const V_PURSUER = 3.0 # Constant speed for the pursuer (must be > V_EVADER)
const W_MAX_PURSUER = pi / 2 # Max turning rate for pursuer (rad/s)
const R_AGENT = 0.25  # Radius for visualization


"""
Calculates the intercept point by solving for the time-to-intercept.
This is the function we are testing.
"""
function calc_intercept_pt(state_evader, state_pursuer)
    p1 = state_evader[1:2]
    psi1 = state_evader[3]
    v1 = state_evader[4]

    p2 = state_pursuer[1:2]
    v2 = V_PURSUER # Pursuer will travel at max speed to intercept

    delta_p = p1 - p2
    v1_vec = v1 * [cos(psi1), sin(psi1)]

    # Quadratic equation coefficients for time-to-intercept 't'
    # (v1^2 - v2^2)t^2 + (2*dot(delta_p, v1_vec))t + norm(delta_p)^2 = 0
    a = dot(v1_vec, v1_vec) - v2^2
    b = 2 * dot(delta_p, v1_vec)
    c = dot(delta_p, delta_p)

    discriminant = b^2 - 4*a*c
    if discriminant < 0
        # This case shouldn't happen if v2 > v1, but it's good practice
        println("Warning: No real solution for intercept.")
        return p1 + v1_vec * 1.0 # Fallback: aim 1 second ahead
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
        println("Warning: No positive time solution for intercept.")
        return p1 + v1_vec * 1.0 # Fallback
    end

    return p1 + v1_vec * t
end


function run_test_simulation()
    println("Starting intercept logic test...")

    # --- Vehicle Initialization ---
    # Evader starts left, moves right
    state1 = [-5.0, 0.0, pi/6, V_EVADER]
    hist_state1 = [state1]

    # Pursuer starts bottom, aims for intercept
    state2 = [0.0, -5.0, 0.0, V_PURSUER]
    hist_state2 = [state2]

    # Calculate the initial, "true" intercept point. The vehicles should meet here.
    initial_intercept_pt = calc_intercept_pt(state1, state2)
    println("Initial predicted intercept point: $(round.(initial_intercept_pt, digits=2))")
    
    # We only need to store the dynamically calculated intercept point for the animation
    dynamic_intercept_hist = [initial_intercept_pt]

    # --- Simulation Loop ---
    for i in 1:N_STEPS
        # --- Update Evader (moves in a straight line) ---
        state1[1] += state1[4] * cos(state1[3]) * DT
        state1[2] += state1[4] * sin(state1[3]) * DT

        # --- Update Pursuer (now with turning dynamics) ---
        intercept_point = calc_intercept_pt(state1, state2)
        
        # Calculate desired heading and heading error
        angle_to_intercept = atan(intercept_point[2] - state2[2], intercept_point[1] - state2[1])
        psi_error = atan(sin(angle_to_intercept - state2[3]), cos(angle_to_intercept - state2[3]))
        
        # Proportional control for angular velocity
        w_cmd = 4.0 * psi_error # The '4.0' is a gain, how fast it tries to correct
        w_clamped = clamp(w_cmd, -W_MAX_PURSUER, W_MAX_PURSUER)

        # Update pursuer's state using the calculated turn rate
        state2[3] += w_clamped * DT # Update heading
        state2[1] += state2[4] * cos(state2[3]) * DT # Update x
        state2[2] += state2[4] * sin(state2[3]) * DT # Update y

        # Store history
        push!(hist_state1, copy(state1))
        push!(hist_state2, copy(state2))
        push!(dynamic_intercept_hist, intercept_point)

        # Check for capture
        if norm(state1[1:2] - state2[1:2]) < R_AGENT * 2
            println("Capture! Distance: $(norm(state1[1:2] - state2[1:2]))")
            break
        end
    end

    println("Test simulation finished.")
    return hist_state1, hist_state2, initial_intercept_pt, dynamic_intercept_hist
end


function plot_test_animation(hist_state1, hist_state2, initial_intercept_pt, dynamic_intercept_hist)
    println("Generating test animation...")
    
    x1_hist = [s[1] for s in hist_state1]; y1_hist = [s[2] for s in hist_state1]
    x2_hist = [s[1] for s in hist_state2]; y2_hist = [s[2] for s in hist_state2]

    theta = 0:0.1:(2*pi+0.1)
    
    # Set plot limits based on trajectory
    xlims = (-12,12)
    ylims = (-6,6)

    anim = @animate for i in 1:length(hist_state1)
        p = plot(x1_hist[1:i], y1_hist[1:i], label="Evader Path", lw=2, c=:blue, aspect_ratio=:equal,
             xlims=xlims, ylims=ylims,
             xlabel="x [m]", ylabel="y [m]", title="Intercept Logic Test (Frame $i)")
        plot!(p, x2_hist[1:i], y2_hist[1:i], label="Pursuer Path", lw=2, c=:red)

        # Plot current vehicle positions
        plot!(p, x1_hist[i] .+ R_AGENT .* cos.(theta), y1_hist[i] .+ R_AGENT .* sin.(theta), seriestype=:shape, fillalpha=0.5, lw=0, c=:blue, label="Evader")
        plot!(p, x2_hist[i] .+ R_AGENT .* cos.(theta), y2_hist[i] .+ R_AGENT .* sin.(theta), seriestype=:shape, fillalpha=0.5, lw=0, c=:red, label="Pursuer")

        # Plot the initial, "true" intercept point as a static target
        scatter!(p, [initial_intercept_pt[1]], [initial_intercept_pt[2]], label="Initial Intercept Pt.", marker=:xcross, markersize=8, color=:black)
        
        # Plot the intercept point as calculated at the current frame
        if i <= length(dynamic_intercept_hist)
            current_pt = dynamic_intercept_hist[i]
            scatter!(p, [current_pt[1]], [current_pt[2]], label="Dynamic Intercept Pt.", marker=:star5, markersize=6, color=:magenta)
        end
    end

    gif_path = "intercept_test1.gif"
    gif(anim, gif_path, fps = 15)
    println("Animation saved to $gif_path")
end

# --- Main Execution ---
hist_state1, hist_state2, initial_intercept_pt, dynamic_intercept_hist = run_test_simulation()
plot_test_animation(hist_state1, hist_state2, initial_intercept_pt, dynamic_intercept_hist)

println("Script finished. Press Enter to exit...")
readline()

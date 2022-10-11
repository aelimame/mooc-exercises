#!/usr/bin/env python
# coding: utf-8

# In[690]:


import numpy as np

# Lateral control

# TODO: write the PID controller using what you've learned in the previous activities

# Note: y_hat will be calculated based on your DeltaPhi() and poseEstimate() functions written previously 

def PIDController(
    v_0, # assume given (by the scenario)
    y_ref, # assume given (by the scenario)
    y_hat, # assume given (by the odometry)
    prev_e_y, # assume given (by the previous iteration of this function)
    prev_int_y, # assume given (by the previous iteration of this function)
    delta_t): # assume given (by the simulator)
    """
    Args:
        v_0 (:double:) linear Duckiebot speed.
        y_ref (:double:) reference lateral pose
        y_hat (:double:) the current estiamted pose along y.
        prev_e_y (:double:) tracking error at previous iteration.
        prev_int_y (:double:) previous integral error term.
        delta_t (:double:) time interval since last call.
    returns:
        v_0 (:double:) linear velocity of the Duckiebot 
        omega (:double:) angular velocity of the Duckiebot
        e_y (:double:) current tracking error (automatically becomes prev_e_y at next iteration).
        e_int_y (:double:) current integral error (automatically becomes prev_int_y at next iteration).
    """
    
    # Tracking error(t): 𝑒(𝑡) = 𝑟𝑡 - 𝑦̂𝑡 = 𝜃𝑟𝑒𝑓t - 𝜃̂^𝑡
    e = y_ref - y_hat #np.random.random()
    
    # Intergral of error(t): e_int_k = 𝑒_𝑖𝑛𝑡_(k-1) + 𝑒_k * Δ𝑡
    e_int = prev_int_y + e * delta_t #np.random.random()
    # anti-windup - preventing the integral error from growing too much
    e_int = max(min(e_int,0.3),-0.3)
    
    # Derivative of error(t): 𝑑𝑒 / 𝑑𝑡 ≃ (𝑒_𝑘 − 𝑒_𝑘−1) / 𝑑𝑡
    e_dr = (e - prev_e_y) / delta_t
    
    # TODO Ziegler-Nichols Closed-Loop Tuning 
    #k_u = 1.66 #
    #T_u = 120 # in m
    #k_p = 0.6 * k_u
    #k_i = 2*k_p / T_u
    #k_d = k_p * T_u / 8

    # Manual tuning
    k_p = 0.5 # sim: 0.5 | dukiebot 2.0
    k_i = 0.016 # sim: 0.016 | dukiebot 0.0
    k_d = 15.0 # sim: 15.0 | dukiebot 5.0
   
    # PID to compute y_rate. This is basically the lateral rate (speed) to get to the objecif y reference.
    y_rate = k_p * e + k_i * e_int + k_d * e_dr
    
    # Convert y_rate to omega (angle rate). We will use the arcsin to convert from linera speeds
    # of y_rate (Opposite side) and v0 (Hypotenuse) to omega.
    MAX_Y_RATE = 1.0
    y_rate_norm = y_rate / v_0    
    y_rate_norm = max(min(y_rate_norm, MAX_Y_RATE),-MAX_Y_RATE) # Limit to 1.0 to be able to get to angle representatio using arcsin
    omega = np.arcsin(y_rate_norm)
    
    # TODO this is probably not needed anymore since we are limiting and normalizing the y_rate
    #MAX_OMEGA_DEG = np.deg2rad(30)
    #MAX_OMEGA_RAD = np.deg2rad(MAX_OMEGA_DEG)
    #omega = max(min(omega, MAX_OMEGA_RAD),-MAX_OMEGA_RAD)
    
    return [v_0, omega], e, e_int


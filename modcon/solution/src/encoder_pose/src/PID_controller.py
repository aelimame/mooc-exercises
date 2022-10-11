#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np

# Heading control

# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your PIDController function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

# TODO: write your PID function for heading control!

def PIDController(v_0, theta_ref, theta_hat, prev_e, prev_int, delta_t):
    """
    Args:
        v_0 (:double:) linear Duckiebot speed (given).
        theta_ref (:double:) reference heading pose
        theta_hat (:double:) the current estiamted theta.
        prev_e (:double:) tracking error at previous iteration.
        prev_int (:double:) previous integral error term.
        delta_t (:double:) time interval since last call.
    returns:
        v_0 (:double:) linear velocity of the Duckiebot 
        omega (:double:) angular velocity of the Duckiebot
        e (:double:) current tracking error (automatically becomes prev_e_y at next iteration).
        e_int (:double:) current integral error (automatically becomes prev_int_y at next iteration).
    """
    
    # TODO: these are random values, you have to implement your own PID controller in here
    #𝑟𝑡 = 𝜃𝑟𝑒𝑓𝑡 = 𝜃𝑟𝑒𝑓
    #𝑦̂𝑡 = 𝜃̂𝑡
    #𝜔𝑡=𝜃˙𝑡=𝑑𝜃𝑡/𝑑𝑡 
    #𝑒(𝑡) = 𝑟𝑡 - 𝑦̂𝑡 = 𝜃𝑟𝑒𝑓t - 𝜃̂𝑡
    #𝑢𝑡=[𝑣0,𝜔]𝑇
    # Control 𝜔 = 𝑢𝑡 = 𝑘𝑝*𝑒(𝑡) + 𝑘𝑖*∫ 𝑡0 𝑒(𝜏) 𝑑𝜏 + 𝑘𝑑*𝑑𝑒𝑡 𝑑𝑡,

    # Tracking error(t): 𝑒(𝑡) = 𝑟𝑡 - 𝑦̂𝑡 = 𝜃𝑟𝑒𝑓t - 𝜃̂^𝑡
    e = theta_ref - theta_hat #np.random.random()
    
    # Intergral of error(t): e_int_k = 𝑒_𝑖𝑛𝑡_(k-1) + 𝑒_k * Δ𝑡
    e_int = prev_int + e * delta_t #np.random.random()
    # anti-windup - preventing the integral error from growing too much
    e_int = max(min(e_int,2),-2)
    
    # Derivative of error(t): 𝑑𝑒 / 𝑑𝑡 ≃ (𝑒_𝑘 − 𝑒_𝑘−1) / 𝑑𝑡
    e_dr = (e - prev_e) / delta_t
    
    k_p = 5.0
    k_i = 0.2 #1.0
    k_d = 0.1 #0.2
    omega = k_p * e + k_i * e_int + k_d * e_dr  # Control 𝜔 = 𝑢𝑡 = 𝑘𝑝*𝑒(𝑡) + 𝑘𝑖*∫ 𝑡0 𝑒(𝜏) 𝑑𝜏 + 𝑘𝑑*𝑑𝑒𝑡 𝑑𝑡,  #np.random.uniform(-8.0, 8.0)
    
    return [v_0, omega], e, e_int

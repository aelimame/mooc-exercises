#!/usr/bin/env python
# coding: utf-8

# In[106]:


# start by importing some things we will need
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy, multivariate_normal
from math import floor, sqrt

# In[110]:


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief

# In[123]:


# Now let's define the predict function

# TODOs:
# - Set a maximum for the diff_i and diff_j to avoid drastic changes due to noise in mesurements?
# - Do this for the generate_measurement_likelihood() function too ?

def histogram_predict(belief, dt, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
        belief_in = belief
        delta_t = dt
        
        # These are maximum "jumps" in the i (delta_d) and j (delta_phi) values
        # We can either ignore the changes that are over these values or just clip them
#        MAX_belief_i_jump = belief.shape[0] * 0.5 # 25% for the moment
#        MAX_belief_j_jump = belief.shape[1] * 0.5 # 25% for the moment
        
        # TODO calculate v and w from ticks using kinematics. You will need  some parameters in the `robot_spec` defined above
        v = 0.0 # replace this with a function that uses the encoder 
        w = 0.0 # replace this with a function that uses the encoder
        r = robot_spec["wheel_radius"]
        l = robot_spec["wheel_baseline"]
        enc_res = robot_spec["encoder_resolution"]
        
        alpha = 2*np.pi/enc_res # rotation per tick in radians 
        
        # Left wheel rotation left_phi
        delta_left_phi = left_encoder_ticks * alpha
        
        # Right wheel rotaion right_phi
        delta_right_phi = right_encoder_ticks * alpha
        
        #print("delta_left_phi: {:0.3}, delta_right_phi: {:0.3}".format(delta_left_phi, delta_right_phi))
        
        # Forward speed v
        v = (r/2) * (delta_right_phi + delta_left_phi) / delta_t
        
        # Angular rate w
        w = (r/(2*l)) * (delta_right_phi - delta_left_phi) / delta_t
        
        #print("v: {:0.3}, w: {:0.3}".format(v, w))
        
        # From v and w we can derivate the and delta_d the delta_phi:
        v_lateral = v * np.sin(w) # This is lateral speed toward left relative to d=0 line
        delta_d = v_lateral * delta_t # Convert to distance, this needs to be substracted from d_t
        
        delta_phi = w * delta_t # Convert to angle from rate, this needs to be added to phi_t
        
        #print("delta_d: {:0.3}, delta_phi: {:0.3}".format(delta_d, delta_phi))
        
        # TODO propagate each centroid forward using the kinematic function
        d_t = grid_spec['d'] + delta_d # replace this with something that adds the new odometry
        phi_t = grid_spec['phi'] + delta_phi # replace this with something that adds the new odometry
        
        #print("d_t.min: {:0.3}, d_t.max: {:0.3}".format(d_t.min(), d_t.max()))
        #print("phi_t.min: {:0.3}, phi_t.max: {:0.3}".format(phi_t.min(), phi_t.max()))
        #print("--")

        p_belief = np.zeros(belief.shape)

        # Accumulate the mass for each cell as a result of the propagation step
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                # If belief[i,j] there was no mass to move in the first place
                if belief[i, j] > 0:
                    # Now check that the centroid of the cell wasn't propagated out of the allowable range
                    if (
                        d_t[i, j] > grid_spec['d_max']
                        or d_t[i, j] < grid_spec['d_min']
                        or phi_t[i, j] < grid_spec['phi_min']
                        or phi_t[i, j] > grid_spec['phi_max']
                    ):
                        continue
                        print("continue")
                    
                    # TODO Now find the cell where the new mass should be added
                    
                    # Use belief[i, j] and d_t[i, j] and divide by grid_spec['delta_d'] to find the increment of i
                    d_diff_i = d_t[i, j] / grid_spec['delta_d']                    

                    # Use belief[i, j] and phi_t[i, j] and divide by grid_spec['delta_phi'] to find the increment of j
                    phi_diff_j = phi_t[i, j] / grid_spec['delta_phi']

                    # Offsets ("middle of the grid") since the indexing starts from 0 not negatives
                    d0_i_offset = grid_spec['d_min'] / grid_spec['delta_d']
                    phi0_j_offset = grid_spec['phi_min'] / grid_spec['delta_phi']
                    
                    # Here we apply either floor() or ceil() depending on delta_d sign, otherwise the 
                    # conversion to indcies my lead to asymmetry
                    if np.sign(delta_d) == 1:
                        i_new = np.ceil(d_diff_i - d0_i_offset)
                    else:
                        i_new = np.floor(d_diff_i - d0_i_offset)
                    i_new = int(i_new)
                    
                    #j_new = int(np.floor(phi_diff_j - phi0_j_offset))
                    if np.sign(delta_phi) == 1:
                        j_new = np.ceil(phi_diff_j - phi0_j_offset)
                    else:
                        j_new = np.floor(phi_diff_j - phi0_j_offset)
                    j_new = int(j_new)

                    # If the new i and j are causing a jump of over the defined MAX thresholds (above)
                    # ignore this update since it is likely just noise
                    #if np.abs(d_diff_i) >= MAX_belief_i_jump or np.abs(phi_diff_j) >= MAX_belief_j_jump:
                    #    #print("out")
                    #    continue

                    # We may still end up outside of bounds because of ceil() and floor()
                    if i_new >= belief.shape[0] or j_new >= belief.shape[1]:
                        #print("out")
                        continue
                    
                    p_belief[i_new, j_new] += belief[i, j]


        # Finally we are going to add some "noise" according to the process model noise
        # This is implemented as a Gaussian blur
        s_belief = np.zeros(belief.shape)
        gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

        if np.sum(s_belief) == 0:
            return belief_in
        belief = s_belief / np.sum(s_belief)
        return belief


# In[86]:


# We will start by doing a little bit of processing on the segments to remove anything that is behing the robot (why would it be behind?)
# or a color not equal to yellow or white

def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments

# In[87]:



def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane
            d_i = -d_i
            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i

# In[88]:


def generate_measurement_likelihood(segments, road_spec, grid_spec):

    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec['d'].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if d_i > grid_spec['d_max'] or d_i < grid_spec['d_min'] or phi_i < grid_spec['phi_min'] or phi_i > grid_spec['phi_max']:
            continue

        # TODO find the cell index that corresponds to the measurement d_i, phi_i
        #i = 1 # replace this
        #j = 1 # replace this
        
        # Offsets ("middle of the grid") since the indexing starts from 0 not negatives
        d0_i_offset = grid_spec['d_min']/grid_spec['delta_d']
        phi0_j_offset = grid_spec['phi_min']/grid_spec['delta_phi']
        
        # Here we apply either floor() or ceil() depending on d_i sign, otherwise the 
        # conversion to indcies my lead to asymmetry 
        i = d_i/grid_spec['delta_d']
        if np.sign(d_i) == 1:
            i = np.ceil(i - d0_i_offset)
        else:
            i = np.floor(i - d0_i_offset)
        i = int(i)
        #i = np.clip(i, 0, measurement_likelihood.shape[0]-1)
        
        j = phi_i/grid_spec['delta_phi']
        if np.sign(phi_i) == 1:
            j = np.ceil(j - phi0_j_offset)
        else:
            j = np.floor(j - phi0_j_offset)
        j = int(j)
        #j = np.clip(j, 0, measurement_likelihood.shape[1]-1)

        
        if i >= measurement_likelihood.shape[0] or j >= measurement_likelihood.shape[1]:
            #print("measurement_likelihood out")
            continue
        
        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


# In[89]:


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        # TODO: combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid probability distribution
        #new_belief = measurement_likelihood # replace this with something that combines the belief and the measurement_likelihood
        
        new_belief = belief * measurement_likelihood
        
        # ** Option 1 **
        # OLD: This was used before the bug causing the cbPredict() in the file filter_lane_node.py to fail.
        # If np.sum(new_belief) == 0 it is likely the belief is way off?
        # We completly reset the measurment as our belief
        # Otherwise normalize new_belief and used it as our belief
        #if np.sum(new_belief) == 0:
        #    belief = measurement_likelihood
        #    belief = belief / np.sum(belief)
        #else:
        #    belief = new_belief / np.sum(new_belief)
        
        # ** Option 2 **
        # This is after the bug fix in filter_lane_node.py
        # Take older belief if new_belief == 0, otherwise take the new but normalized (it is likely the measurement is just noisy so ignore it?)
        if np.sum(new_belief) != 0:
            belief = new_belief / np.sum(new_belief)
            
        # Apply gaussian filter, this helps for stabilty in the SIM
        filtered_belief = np.zeros(belief.shape)
        # Robot Home
        gaussian_filter(belief, sigma=[0.6, 0.9], output=filtered_belief, mode="constant")
        # SIM
        #gaussian_filter(belief, sigma=[1.2, 1.4], output=filtered_belief, mode="constant")
        
        if np.sum(filtered_belief) != 0:
            belief = filtered_belief / np.sum(filtered_belief)

    return (measurement_likelihood, belief)


import numpy as np

lanes_locations = np.array([5, 3, 1, -1, -3, -5]) * 2
# State: X, Y, PSI, DELTA, V

def lane_index_to_lane_boundary(index):
    lanes_locations = np.array([5, 3, 1, -1, -3, -5])# * 2
    return np.array([lanes_locations[index], lanes_locations[index+1]])

def get_desired_lane_index(t):
    lane_times = np.array([0,3,4,10])
    lane_ids = np.array([2,3,4,4])
    # Create mask: True where t >= lower_bounds
    mask = t >= lane_times
    # Find first True from the right (i.e., max valid index)
    valid_indices = np.where(mask, np.arange(len(lane_times)), -1)
    selected_index = np.max(valid_indices)  # max because we want highest bound satisfied
    return lane_ids[selected_index]

def reward_within_boundary(state, boundary):
    return np.min( np.array([boundary[0]-state[1], state[1]-boundary[1]]) )

def reward_outside_boundary(state, boundary):
    return -3.0

def compute_reward(t, state, control):
    desired_lane_index = get_desired_lane_index(t)
    boundaries = lane_index_to_lane_boundary(desired_lane_index)
    within_boundary = (state[1] < boundaries[0]) and (state[0] > boundaries[1])
    if within_boundary:
        reward = reward_within_boundary(state, boundaries)
    else:
        reward = reward_outside_boundary(state, boundaries)
    reward = reward - 0.1 * ( state[4] - 0.5 )*( state[4] - 0.5 ) - 0.1 * state[2]*state[2]

    reward = state[4]*state[4] * np.cos(state[2]) - state[2] * state[2]

    speed_error = np.max(np.array([(state[4]-5)*(state[4]-5), 0.01]))
    
    # reward = 0.01 * np.cos(state[2]) / speed_error #- 0.01 * state[2] * state[2]
    # reward = 0.01 / speed_error # - 0.01 * state[2] * state[2]

    if speed_error < 3:
        reward = 2
    elif speed_error < 0.3:
        reward = 4
    else:
        reward = - speed_error
    heading_mod = state[2] * state[2]
    if heading_mod > np.pi/6:
        reward = -10
    else:
        reward += - 0.001 * state[2] * state[2]



    return reward
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
import jax 

lanes_locations = np.array([5, 3, 1, -1, -3, -5]) * 2
# State: X, Y, PSI, DELTA, V

@jit
def lane_index_to_lane_boundary(index):
    lanes_locations = np.array([5, 3, 1, -1, -3, -5])# * 2
    return jnp.array([lax.dynamic_index_in_dim(lanes_locations, index, axis=0, keepdims=False), lax.dynamic_index_in_dim(lanes_locations, index+1, axis=0, keepdims=False)])

@jit
def get_desired_lane_index(t):

    lane_times = jnp.array([0,3,4,10])
    lane_ids = jnp.array([2,3,4,4])
    # Create mask: True where t >= lower_bounds
    mask = t >= lane_times
    # Find first True from the right (i.e., max valid index)
    valid_indices = jnp.where(mask, jnp.arange(len(lane_times)), -1)
    selected_index = jnp.max(valid_indices)  # max because we want highest bound satisfied
    # Step 4: use lax to safely index into values
    return lax.dynamic_index_in_dim(lane_ids, selected_index, axis=0, keepdims=False)


@jit
def reward_within_boundary(state, boundary):
    return jnp.min( jnp.array([boundary[0]-state[1], state[1]-boundary[1]]) )
@jit
def reward_outside_boundary(state, boundary):
    return -3.0
@jit
def reward(t, state, control):
    desired_lane_index = get_desired_lane_index(t)
    boundaries = lane_index_to_lane_boundary(desired_lane_index)
    within_boundary = jnp.logical_and(state[1] < boundaries[0], state[0] > boundaries[1])
    reward = lax.cond( within_boundary, reward_within_boundary, reward_outside_boundary, state, boundaries)
    reward = reward - 0.1 * ( state[4] - 0.5 )*( state[4] - 0.5 ) - 0.1 * state[2]*state[2]
    return reward

def main():
    print(get_desired_lane_index(2.5))
    print(reward(0.5, jnp.array([1, 0.8, 0, 0, 0]), jnp.array([2,3])))

if __name__=="__main__":
    main()
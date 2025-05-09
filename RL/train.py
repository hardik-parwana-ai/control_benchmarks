import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

def lane_index_to_lane_boundary(lanes_locations, index):
    return lanes_locations[index, index+1]

@jit
def reward(t, state, control, lanes_locations):
    if t<3:
        desired_lane_index = 2
    elif t<4:
        desired_lane_index = 3
    elif t<10:
        desired_lane_index = 4
    boundaries = lane_index_to_lane_boundary(lanes_locations, desired_lane_index)
    within_boundary = state[1] < boundaries[1] and state[0] > boundaries[0]
    if within_boundary:
        reward = jnp.min( [boundaries[1]-state[1], state[0]-boundaries[0]] )
    else:
        reward = -3
    return reward # 1, 0 sparse reward

# implement buffer and rollout trajectories
# Try DQN first??
#




def main():

    state = jnp.array([0,0,0,0,0])
    fig, ax = plt.subplots()
    robot = bicycle(state)





if __name__ == "__main__":
    main()

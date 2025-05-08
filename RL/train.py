import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

class bicycle:

    def __init__(self,init_state = jnp.array([0,0,0,0,0]), dt = 0.05, ax=None):
        self.X = init_state # X, Y, PSI, DELTA, V
        self.dt = dt
        self.ax = ax
        self.body = self.ax.scatter([], [], c='g', alpha=1.0)

    @staticmethod
    @jit
    def step(state, control, dt):
        next_state = state + jnp.array([
            state[4] * jnp.cos(state[2]),
            state[4] * jnp.sin(state[2]),
            state[3],
            control[1],
            control[0]
        ]) * dt

        # put steering constraints

    def render(self,state):
        self.body.set_offsets( state[0], state[1] )

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

def main():

    state = jnp.array([0,0,0,0,0])
    fig, ax = plt.subplots()
    robot = bicycle(state)


    lanes_locations = jnp.array([5, 3, 1, -1, -3, -5])





if __name__ == "__main__":
    main()

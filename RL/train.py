import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

def main():

    state = jnp.array([0,0,0,0,0])
    fig, ax = plt.subplots()
    robot = bicycle(state)

    vehicle = f150(ax=ax)
    state = np.array([0,0,0,0,0])



if __name__ == "__main__":
    main()

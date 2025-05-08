import matplotlib.pyplot as plt
import numpy as np

# Sample data for 9 small plots
num_plots = 9
data = [np.random.rand(10) for _ in range(num_plots)]

# Create the grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(10, 8))
axes = axes.flatten()

# Plotting each subplot
for i, ax in enumerate(axes):
    ax.plot(data[i])
    ax.set_title(f'Plot {i}')
    ax._my_index = i  # attach custom attribute to track plot index

# Define click handler
def on_click(event):
    if event.inaxes:
        index = getattr(event.inaxes, '_my_index', None)
        if index is not None:
            # Create new figure without calling plt.show() again
            new_fig, new_ax = plt.subplots(figsize=(6, 4))
            new_ax.plot(data[index])
            new_ax.set_title(f'Zoomed Plot {index}')
            new_fig.canvas.draw()  # draw the figure
            new_fig.show()  # display it without re-entering the main loop

# Connect the click event to handler
fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()

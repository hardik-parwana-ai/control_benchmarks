import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate sample data
np.random.seed(0)
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)
color_values = x + y + z  # Just some values to map to color

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make scatter plot
sc = ax.scatter(x, y, z, c=color_values, cmap='viridis')

# Add colorbar
cb = fig.colorbar(sc, ax=ax, shrink=0.6)
cb.set_label('Color scale')

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Scatter Plot with Colorbar")

plt.show()

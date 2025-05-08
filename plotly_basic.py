import plotly.graph_objects as go
import numpy as np

# Sample data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Plotly surface plot
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

fig.update_layout(
    title="3D Surface Plot",
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.show()



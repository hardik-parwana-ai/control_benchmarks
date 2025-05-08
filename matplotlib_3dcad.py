import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh

# Load your CAD model (STL format)
# robot_mesh = mesh.Mesh.from_file('simplify_f150.obj')  # path to your STL file
# robot_mesh = mesh.Mesh.from_file('f150_3dless_com_simplified.stl')  # path to your STL file
robot_mesh = trimesh.load('f150_3dless_com_simplified.stl')
vertices = robot_mesh.vertices
faces = robot_mesh.faces

# Extract vertices
def get_patch_data(mesh_obj):
    x = mesh_obj.vectors[:, :, 0]
    y = mesh_obj.vectors[:, :, 1]
    z = mesh_obj.vectors[:, :, 2]
    return x, y, z

# Animation update function
def update(num, mesh_obj, plot):
    translation = np.array([0.05*num, 0, 0])  # Move along X axis
    mesh_copy = mesh_obj.copy()
    mesh_copy.translate(translation)

    x, y, z = get_patch_data(mesh_copy)
    plot[0].remove()  # Remove old plot
    plot[0] = ax.plot_surface(x, y, z, color='cyan', edgecolor='k', rstride=1, cstride=1, alpha=0.7)

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# x, y, z = get_patch_data(robot_mesh)
# plot = [ax.plot_surface(x, y, z, color='cyan', edgecolor='k', rstride=1, cstride=1, alpha=0.7)]

mesh_collection = Poly3DCollection(vertices[faces], facecolors='skyblue', edgecolors='k', alpha=0.9)
ax.add_collection3d(mesh_collection)

# Setup animation
# ani = animation.FuncAnimation(fig, update, frames=50, fargs=(robot_mesh, plot), interval=100)

plt.show()

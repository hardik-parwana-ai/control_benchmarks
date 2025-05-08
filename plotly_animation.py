import numpy as np
import plotly.graph_objects as go

class PlotlyF150:
    def __init__(self, facecolor="rgba(0,0,0,0.3)", alpha=0.3):
        self.width = 2
        self.length = 4
        self.wheel_length = 0.6
        self.wheel_width = 0.3
        self.facecolor = facecolor
        self.alpha = alpha

        self.wheel1_center = np.array([self.length / 3, 2 * self.width / 5])
        self.wheel2_center = np.array([self.length / 3, -2 * self.width / 5])
        self.wheel3_center = np.array([-self.length / 3, -2 * self.width / 5])
        self.wheel4_center = np.array([-self.length / 3, 2 * self.width / 5])

        self.pos = np.array([0, 0])
        self.yaw = 0
        self.steering = 0

        self.base_front_points = np.array([
            [0.5 * self.length / 2, -self.width / 2],
            [self.length / 2, -self.width / 2],
            [self.length / 2, self.width / 2],
            [0.5 * self.length / 2, self.width / 2],
        ])
        self.base_rear_points = np.array([
            [-self.length / 2, -self.width / 2],
            [0.5 * self.length / 2, -self.width / 2],
            [0.5 * self.length / 2, self.width / 2],
            [-self.length / 2, self.width / 2],
        ])

        self.wheel_points = np.array([
            [-self.wheel_length / 2, -self.wheel_width / 2],
            [self.wheel_length / 2, -self.wheel_width / 2],
            [self.wheel_length / 2, self.wheel_width / 2],
            [-self.wheel_length / 2, self.wheel_width / 2],
        ])

    def rot_mat(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

    def transform_points(self, points, rotation, translation):
        return (rotation @ points.T).T + translation

    def render(self, X):
        self.pos = np.array([X[0], X[1]])
        self.yaw = X[2]
        self.steering = X[3]

        R = self.rot_mat(self.yaw)
        R_steer = self.rot_mat(self.steering)

        elements = []

        front_points = self.transform_points(self.base_front_points, R, self.pos)
        rear_points = self.transform_points(self.base_rear_points, R, self.pos)
        elements.append(self.polygon_trace(front_points, 'green'))
        elements.append(self.polygon_trace(rear_points, 'red'))

        wheels = [self.wheel1_center, self.wheel2_center, self.wheel3_center, self.wheel4_center]
        for i, center in enumerate(wheels):
            wheel_rot = R_steer if i < 2 else np.eye(2)
            wheel_shape = self.transform_points(self.wheel_points, wheel_rot, center)
            wheel_shape = self.transform_points(wheel_shape, R, self.pos)
            elements.append(self.polygon_trace(wheel_shape, 'black'))

        return elements

    def polygon_trace(self, points, color):
        return go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="lines",
            fill="toself",
            line=dict(color=color),
            fillcolor=color,
            showlegend=False
        )

def simulate_trajectory_plotly(ref, traj):
    f150 = PlotlyF150()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref[0], y=ref[1], mode="lines", name="Reference", line=dict(color="black")))

    first_frame = f150.render(traj[0])
    for trace in first_frame:
        fig.add_trace(trace)

    frames = []
    for state in traj:
        vehicle_traces = f150.render(state)
        frames.append(go.Frame(data=vehicle_traces))

    fig.update(frames=frames)

    fig.update_layout(
        xaxis=dict(range=[-5, 20]),
        yaxis=dict(range=[-5, 20]),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )],
        title="F150 Trajectory Animation"
    )

    fig.show()

if __name__ == "__main__":
    # Sample reference and trajectory data
    t = np.linspace(0, 10, 50)
    x = t
    y = np.sin(t)
    yaw = np.arctan(np.cos(t))
    steering = 0.3 * np.sin(0.5 * t)

    ref = [x, y]
    traj = np.vstack((x, y, yaw, steering)).T

    simulate_trajectory_plotly(ref, traj)

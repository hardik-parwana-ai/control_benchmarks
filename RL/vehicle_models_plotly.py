import numpy as np
import plotly.graph_objects as go

import jax.numpy as jnp
from jax import jit

class F150:
    def __init__(self, init_state=np.array([0,0,0,0,0]), facecolor="k", alpha=0.3):
        # chassis
        self.width = 2
        self.length = 4
        # wheels
        self.wheel_length = 0.6
        self.wheel_width = 0.3
        self.wheel1_center = np.array([self.length / 3, 2 * self.width / 5]).reshape(
            -1, 1
        )
        self.wheel2_center = np.array([self.length / 3, -2 * self.width / 5]).reshape(
            -1, 1
        )
        self.wheel3_center = np.array([-self.length / 3, -2 * self.width / 5]).reshape(
            -1, 1
        )
        self.wheel4_center = np.array([-self.length / 3, 2 * self.width / 5]).reshape(
            -1, 1
        )
        self.pos = init_state[0:2] #np.array([0, 0]).reshape(-1, 1)
        self.yaw = init_state[2]#0
        self.steering = init_state[3] #0

        # Vehcile chassis
        self.base_points = np.array(
            [
                [-self.length / 2, -self.width / 2],
                [self.length / 2, -self.width / 2],
                [self.length / 2, self.width / 2],
                [-self.length / 2, self.width / 2],
            ]
        )
        self.base_front_points = np.array(
            [
                [0.5 * self.length / 2, -self.width / 2],
                [self.length / 2, -self.width / 2],
                [self.length / 2, self.width / 2],
                [0.5 * self.length / 2, self.width / 2],
            ]
        )
        self.base_rear_points = np.array(
            [
                [-self.length / 2, -self.width / 2],
                [0.5 * self.length / 2, -self.width / 2],
                [0.5 * self.length / 2, self.width / 2],
                [-self.length / 2, self.width / 2],
            ]
        )

        self.wheel_points = np.array(
            [
                [-self.wheel_length / 2, -self.wheel_width / 2],
                [self.wheel_length / 2, -self.wheel_width / 2],
                [self.wheel_length / 2, self.wheel_width / 2],
                [-self.wheel_length / 2, self.wheel_width / 2],
            ]
        )

    def rot_mat(self, theta):
        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def transform_points(self, points, rotation, translation):
        # pdb.set_trace()
        return (rotation @ points.T) + translation.reshape(-1, 1)

    def polygon_trace(self, points, color, opacity=1.0):
        # print(points[0, :])
        return go.Scatter(
            x=points[0, :],
            y=points[1, :],
            mode="lines",
            fill="toself",
            line=dict(color=color),
            fillcolor=color,
            showlegend=False,
            opacity=opacity,
        )

    def render(self, X):
        self.pos = np.array([X[0], X[1]])
        self.yaw = X[2]
        self.steering = X[3]

        R = self.rot_mat(self.yaw)
        R_steer = self.rot_mat(self.steering)

        elements = []

        front_points = self.transform_points(self.base_front_points, R, self.pos)
        rear_points = self.transform_points(self.base_rear_points, R, self.pos)
        elements.append(self.polygon_trace(front_points, "green", opacity=0.5))
        elements.append(self.polygon_trace(rear_points, "red", opacity=0.5))

        wheels = [
            self.wheel1_center,
            self.wheel2_center,
            self.wheel3_center,
            self.wheel4_center,
        ]
        for i, center in enumerate(wheels):
            # pdb.set_trace()
            wheel_rot = R_steer if i < 2 else np.eye(2)
            wheel_shape = self.transform_points(
                self.wheel_points, wheel_rot, np.zeros(2)
            )
            wheel_shape = self.transform_points(self.wheel_points, wheel_rot, center)
            wheel_shape = self.transform_points(wheel_shape.T, R, self.pos)
            elements.append(self.polygon_trace(wheel_shape, "black"))

        return elements

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
        return next_state


def main():
    state = np.array([0,0,0,0,0])
    tf = 5
    dt = 0.05
    t = 0 

    f150 = F150(init_state=state)

    fig = go.Figure()
    
    
    trajectory = np.array(list(zip(*traj)))

    first_frame = f150.render(trajectory[0])
    for trace in first_frame:
        fig.add_trace(trace)

    frames = []
    for state in trajectory:
        vehicle_traces = f150.render(state)
        frames.append(go.Frame(data=vehicle_traces))

    # Add empty traces for vehicle parts to be updated by animation
    for _ in range(len(frames[0].data)):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                fill="toself",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            )
        )

    fig.update(frames=frames)

    fig.update_layout(
        xaxis=dict(range=[-5, 20]),
        yaxis=dict(range=[-5, 20]),
        width=800,  # Set the width in pixels
        height=800,  # Set the height in pixels
        yaxis_scaleanchor="x",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                ],
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": str(i),
                        "method": "animate",
                    }
                    for i, f in enumerate(frames)
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "xanchor": "left",
                "y": -0.1,
                "yanchor": "top",
            }
        ],
        title="F150 Trajectory Animation",
    )

    fig.show()

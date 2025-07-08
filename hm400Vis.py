import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


class HM400:
    def __init__(self, ax=None, param_file=None, facecolor="k", alpha=0.3, dt=0.05):
        # chassis
        self.dt = dt
        self.width = 2.690
        self.length_cabin = 4.335
        self.length_dump_bed = 5.667
        self.length_link = 1103
        self.hitch_to_cabin_axle = 1.350
        self.hitch_to_dump_bed_rear_axle = 4.970
        self.hitch_to_dump_bed_front_axle = 3.00
        self.hitch_to_cabin_center = 2.1675
        self.hitch_to_dump_bed_center = 3.9365
        self.dump_bed_rear_axle_to_dump_bed_center = (
            self.hitch_to_dump_bed_rear_axle - self.hitch_to_dump_bed_center
        )
        self.hitch_to_cabin_rear = self.hitch_to_cabin_center - self.length_cabin / 2
        self.hitch_to_dump_bed_front = (
            self.hitch_to_dump_bed_center - self.length_dump_bed / 2
        )
        cabin_rear_wrt_hitch = np.array([self.hitch_to_cabin_rear, 0]).reshape(-1, 1)
        dump_bed_wrt_hitch = np.array([-self.hitch_to_dump_bed_front, 0]).reshape(-1, 1)
        self.link_base_wrt_hitch = np.append(
            cabin_rear_wrt_hitch, dump_bed_wrt_hitch, axis=1
        )

        # center at rear axle of rear vehicle
        # wheel numbering start from
        # wheels
        self.wheel_length = 0.6
        self.wheel_width = 0.3

        # describe centers w.r.t hitch joint
        self.wheel1_center_wrt_hitch = np.array(
            [self.hitch_to_cabin_axle, self.width / 2]
        ).reshape(-1, 1)
        self.wheel2_center_wrt_hitch = np.array(
            [self.hitch_to_cabin_axle, -self.width / 2]
        ).reshape(-1, 1)
        self.wheel3_center_wrt_hitch = np.array(
            [-self.hitch_to_dump_bed_front_axle, self.width / 2]
        ).reshape(-1, 1)
        self.wheel4_center_wrt_hitch = np.array(
            [-self.hitch_to_dump_bed_front_axle, -self.width / 2]
        ).reshape(-1, 1)
        self.wheel5_center_wrt_hitch = np.array(
            [-self.hitch_to_dump_bed_rear_axle, self.width / 2]
        ).reshape(-1, 1)
        self.wheel6_center_wrt_hitch = np.array(
            [-self.hitch_to_dump_bed_rear_axle, -self.width / 2]
        ).reshape(-1, 1)
        self.pos = np.array([0, 0]).reshape(-1, 1)  # center of rear axle
        self.yaw_rear = 0
        self.yaw_front = 0
        self.steering = 0

        # Vehcile chassis
        self.cabin_basis_points = np.array(
            [
                [-self.length_cabin / 2, -self.width / 2],
                [self.length_cabin / 2, -self.width / 2],
                [self.length_cabin / 2, self.width / 2],
                [-self.length_cabin / 2, self.width / 2],
            ]
        ).T
        self.dump_bed_basis_points = np.array(
            [
                [-self.length_dump_bed / 2, -self.width / 2],
                [self.length_dump_bed / 2, -self.width / 2],
                [self.length_dump_bed / 2, self.width / 2],
                [-self.length_dump_bed / 2, self.width / 2],
            ]
        ).T
        self.wheel_basis_points = np.array(
            [
                [-self.wheel_length / 2, -self.wheel_width / 2],
                [self.wheel_length / 2, -self.wheel_width / 2],
                [self.wheel_length / 2, self.wheel_width / 2],
                [-self.wheel_length / 2, self.wheel_width / 2],
            ]
        ).T

    def rot_mat(self, theta):
        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def transform_points(self, points, rotation, translation):
        return (rotation @ points) + translation.reshape(-1, 1)

    def compute_transformed_points(self, X):
        hitch_wrt_dump_bed = np.array([self.hitch_to_dump_bed_rear_axle, 0]).reshape(
            -1, 1
        )
        hitch_joint_position = self.transform_points(
            hitch_wrt_dump_bed, self.rot_mat(self.yaw), self.rear_pos
        )

        # rear
        dump_bed_center_position = self.transform_points(
            np.array([self.dump_bed_rear_axle_to_dump_bed_center, 0.0]).reshape(-1, 1),
            self.rot_mat(self.yaw),
            self.rear_pos,
        )
        dump_bed_points = self.transform_points(
            self.dump_bed_basis_points, self.rot_mat(self.yaw), dump_bed_center_position
        )

        # front
        cabin_center_wrt_hitch = np.array([self.hitch_to_cabin_center, 0.0]).reshape(
            -1, 1
        )
        cabin_center_position = self.transform_points(
            cabin_center_wrt_hitch, self.rot_mat(self.yaw_front), hitch_joint_position
        )
        cabin_points = self.transform_points(
            self.cabin_basis_points, self.rot_mat(self.yaw_front), cabin_center_position
        )

        # link points
        link_points = self.transform_points(
            self.link_base_wrt_hitch, self.rot_mat(self.yaw), hitch_joint_position
        )

        # Wheel 1 - 6
        wheel1_center_position = self.transform_points(
            self.wheel1_center_wrt_hitch,
            self.rot_mat(self.yaw_front),
            hitch_joint_position,
        )
        wheel1_points = self.transform_points(
            self.wheel_basis_points,
            self.rot_mat(self.yaw_front),
            wheel1_center_position,
        )

        wheel2_center_position = self.transform_points(
            self.wheel2_center_wrt_hitch,
            self.rot_mat(self.yaw_front),
            hitch_joint_position,
        )
        wheel2_points = self.transform_points(
            self.wheel_basis_points,
            self.rot_mat(self.yaw_front),
            wheel2_center_position,
        )

        wheel3_center_position = self.transform_points(
            self.wheel3_center_wrt_hitch, self.rot_mat(self.yaw), hitch_joint_position
        )
        wheel3_points = self.transform_points(
            self.wheel_basis_points, self.rot_mat(self.yaw), wheel3_center_position
        )

        wheel4_center_position = self.transform_points(
            self.wheel4_center_wrt_hitch, self.rot_mat(self.yaw), hitch_joint_position
        )
        wheel4_points = self.transform_points(
            self.wheel_basis_points, self.rot_mat(self.yaw), wheel4_center_position
        )

        wheel5_center_position = self.transform_points(
            self.wheel5_center_wrt_hitch, self.rot_mat(self.yaw), hitch_joint_position
        )
        wheel5_points = self.transform_points(
            self.wheel_basis_points, self.rot_mat(self.yaw), wheel5_center_position
        )

        wheel6_center_position = self.transform_points(
            self.wheel6_center_wrt_hitch, self.rot_mat(self.yaw), hitch_joint_position
        )
        wheel6_points = self.transform_points(
            self.wheel_basis_points, self.rot_mat(self.yaw), wheel6_center_position
        )

        return (
            cabin_points,
            dump_bed_points,
            link_points,
            [
                wheel1_points,
                wheel2_points,
                wheel3_points,
                wheel4_points,
                wheel5_points,
                wheel6_points,
            ],
        )

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
        # X = x, y, psi, delta, v
        self.pos = np.array([X[0], X[1]])
        self.yaw = X[2]
        self.steering = X[3]
        self.yaw_front = self.yaw + self.steering
        self.rear_pos = np.array([X[0], X[1]])

        elements = []
        cabin_points, dump_bed_points, link_points, wheel_points = (
            self.compute_transformed_points(X)
        )
        elements.append(self.polygon_trace(cabin_points, "green", opacity=0.5))
        elements.append(self.polygon_trace(dump_bed_points, "red", opacity=0.5))
        for i in range(6):
            elements.append(self.polygon_trace(wheel_points[i], "black", opacity=0.5))
        elements.append(
            go.Scatter(
                x=link_points[0, :],
                y=link_points[1, :],
                mode="lines",
                line=dict(color="black"),
            )
        )
        return elements

    def step(self, X, U):
        # x = X[0]
        # y = X[1]
        psi = X[2]
        steering = X[3]
        V = X[4]
        steering_rate = U[0]
        accel = U[1]
        xdot = np.array(
            [
                V * np.cos(psi),
                V * np.sin(psi),
                (V * np.sin(steering) - self.hitch_to_cabin_axle * steering_rate)
                / (
                    self.hitch_to_cabin_axle
                    + self.hitch_to_dump_bed_rear_axle * np.cos(steering)
                ),
                steering_rate,
                accel,
            ]
        )
        self.X = X + xdot * self.dt
        return self.X


def animate_trajectory(ref=None, traj=None):
    file_name = "params.txt"

    hm400 = HM400(param_file=file_name)

    fig = go.Figure()
    if ref is not None:
        ref_trace = go.Scatter(
            x=ref[0], y=ref[1], mode="lines", name="Reference", line=dict(color="black")
        )
        fig.add_trace(ref_trace)
    vehicle_path_trace = go.Scatter(
        x=traj[0, :],
        y=traj[1, :],
        mode="lines",
        name="Vehicle",
        line=dict(color="green"),
    )
    fig.add_trace(vehicle_path_trace)
    # trajectory = np.array(list(zip(*traj)))

    first_frame = hm400.render(traj[:, 0])
    for trace in first_frame:
        fig.add_trace(trace)

    frames = []
    for i in range(traj.shape[1]):
        vehicle_traces = hm400.render(traj[:, i])
        if ref is not None:
            frames.append(
                go.Frame(data=[ref_trace, vehicle_path_trace] + vehicle_traces)
            )
        else:
            frames.append(go.Frame(data=[vehicle_path_trace] + vehicle_traces))

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
        xaxis=dict(range=[-20, 25]),
        yaxis=dict(range=[-20, 25]),
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
        title="HM400 Trajectory Animation",
    )

    fig.show()


def main():
    X = np.array([0, 0, 0, 0, 0])
    dt = 0.05
    tf = 20
    t = 0
    traj = X.reshape(-1, 1)
    vehicle = HM400(dt=dt)

    while t < tf:
        if t < tf / 2:
            action = np.array([0.1, 0.4])
        else:
            action = np.array([-0.1, 0.4])
        X = vehicle.step(X, action)
        traj = np.append(traj, X.reshape(-1, 1), axis=1)
        t += dt
    animate_trajectory(traj=traj)


if __name__ == "__main__":
    main()

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jax import grad, jit, lax
# from plotly_utils import plotly_update_menus

import plotly.io as pio
pio.renderers.default = 'browser'


class HM400:
    def __init__(
        self,
        ax=None,
        param_file=None,
        facecolor="k",
        alpha=0.3,
        dt=0.05,
        pure_pursuit_mode="rear",
    ):
        # chassis
        self.pure_pursuit_mode = pure_pursuit_mode
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
        # cabin_rear_wrt_hitch = np.array([self.hitch_to_cabin_rear, 0]).reshape(-1, 1)
        # dump_bed_wrt_hitch = np.array([-self.hitch_to_dump_bed_front, 0]).reshape(-1, 1)
        cabin_rear_wrt_hitch = np.array([self.hitch_to_cabin_axle, 0]).reshape(-1, 1)
        dump_bed_wrt_hitch = np.array([-self.hitch_to_dump_bed_rear_axle, 0]).reshape(
            -1, 1
        )
        self.link_base_wrt_hitch = np.concatenate(
            (
                dump_bed_wrt_hitch,
                np.zeros((2, 1)),
                cabin_rear_wrt_hitch,
            ),
            axis=1,
        )
        self.steering_bound = np.pi / 6
        self.steering_rate_bound = 0.3  # 2.0  # 0.6 #0.6
        self.acc_bound = 1.2
        self.lookahead_distance = 30

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
        yaw_rot_mat = self.rot_mat(self.yaw)
        yaw_front_rot_mat = self.rot_mat(self.yaw_front)
        hitch_wrt_dump_bed = np.array([self.hitch_to_dump_bed_rear_axle, 0]).reshape(
            -1, 1
        )
        hitch_joint_position = self.transform_points(
            hitch_wrt_dump_bed, yaw_rot_mat, self.rear_pos
        )

        # rear
        dump_bed_center_position = self.transform_points(
            np.array([self.dump_bed_rear_axle_to_dump_bed_center, 0.0]).reshape(-1, 1),
            yaw_rot_mat,
            self.rear_pos,
        )
        dump_bed_points = self.transform_points(
            self.dump_bed_basis_points, yaw_rot_mat, dump_bed_center_position
        )

        # front
        cabin_center_wrt_hitch = np.array([self.hitch_to_cabin_center, 0.0]).reshape(
            -1, 1
        )
        cabin_center_position = self.transform_points(
            cabin_center_wrt_hitch, yaw_front_rot_mat, hitch_joint_position
        )
        cabin_points = self.transform_points(
            self.cabin_basis_points, yaw_front_rot_mat, cabin_center_position
        )

        # link points
        link_points = np.append(
            self.transform_points(
                self.link_base_wrt_hitch[:, 0:2], yaw_rot_mat, hitch_joint_position
            ),
            self.transform_points(
                self.link_base_wrt_hitch[:, 1:3],
                yaw_front_rot_mat,
                hitch_joint_position,
            ),
            axis=1,
        )

        # Wheel 1 - 6
        wheel1_center_position = self.transform_points(
            self.wheel1_center_wrt_hitch,
            yaw_front_rot_mat,
            hitch_joint_position,
        )
        wheel1_points = self.transform_points(
            self.wheel_basis_points,
            yaw_front_rot_mat,
            wheel1_center_position,
        )

        wheel2_center_position = self.transform_points(
            self.wheel2_center_wrt_hitch,
            yaw_front_rot_mat,
            hitch_joint_position,
        )
        wheel2_points = self.transform_points(
            self.wheel_basis_points,
            yaw_front_rot_mat,
            wheel2_center_position,
        )

        wheel3_center_position = self.transform_points(
            self.wheel3_center_wrt_hitch, yaw_rot_mat, hitch_joint_position
        )
        wheel3_points = self.transform_points(
            self.wheel_basis_points, yaw_rot_mat, wheel3_center_position
        )

        wheel4_center_position = self.transform_points(
            self.wheel4_center_wrt_hitch, yaw_rot_mat, hitch_joint_position
        )
        wheel4_points = self.transform_points(
            self.wheel_basis_points, yaw_rot_mat, wheel4_center_position
        )

        wheel5_center_position = self.transform_points(
            self.wheel5_center_wrt_hitch, yaw_rot_mat, hitch_joint_position
        )
        wheel5_points = self.transform_points(
            self.wheel_basis_points, yaw_rot_mat, wheel5_center_position
        )

        wheel6_center_position = self.transform_points(
            self.wheel6_center_wrt_hitch, yaw_rot_mat, hitch_joint_position
        )
        wheel6_points = self.transform_points(
            self.wheel_basis_points, yaw_rot_mat, wheel6_center_position
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

    def render(self, X, Xd=None):
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
                line=dict(color="black", width=5),
            )
        )
        if Xd is not None:
            elements.append(
                go.Scatter(
                    x=Xd[[0]],
                    y=Xd[[1]],
                    mode="markers",
                    line=dict(color="black"),
                    name="Chosen reference",
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

    @partial(jit, static_argnums=(0,))
    def step_rear_jax(self, X, U, dt):
        psi = X[2]
        steering = X[3]
        V = X[4]
        steering_rate = U[0]
        accel = U[1]

        steering_rate_min = (-self.steering_bound - steering) / self.dt
        steering_rate_max = (self.steering_bound - steering) / self.dt
        steering_rate = jnp.clip(steering_rate, steering_rate_min, steering_rate_max)

        xdot = jnp.array(
            [
                V * jnp.cos(psi),
                V * jnp.sin(psi),
                (V * jnp.sin(steering) - self.hitch_to_cabin_axle * steering_rate)
                / (
                    self.hitch_to_cabin_axle
                    + self.hitch_to_dump_bed_rear_axle * jnp.cos(steering)
                ),
                steering_rate,
                accel,
            ]
        )
        X_next = X + xdot * dt
        X_next = X_next.at[2].set(self.wrap_angle(X_next[2]))
        return X_next

    @partial(jit, static_argnums=(0,))
    def wrap_angle(self, angle):
        return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))

    @partial(jit, static_argnums=(0,))
    def step_front_jax(self, X, U, dt):
        psi = X[2]
        steering = X[3]
        V = X[4]
        steering_rate = U[0]
        accel = U[1]

        steering_rate_min = (-self.steering_bound - steering) / self.dt
        steering_rate_max = (self.steering_bound - steering) / self.dt
        steering_rate = jnp.clip(steering_rate, steering_rate_min, steering_rate_max)

        xdot = jnp.array(
            [
                V * jnp.cos(psi),
                V * jnp.sin(psi),
                -(
                    V * jnp.sin(steering)
                    + self.hitch_to_dump_bed_rear_axle * steering_rate
                )
                / (
                    self.hitch_to_cabin_axle * jnp.cos(steering)
                    + self.hitch_to_dump_bed_rear_axle
                ),
                steering_rate,
                accel,
            ]
        )
        X_next = X + xdot * dt
        return X_next

    @partial(jit, static_argnums=(0,))
    def func(self, delta, C):
        l1 = self.hitch_to_dump_bed_rear_axle
        l2 = self.hitch_to_cabin_axle
        return (l1 + l2 / jnp.cos(delta)) * (
            (l1**2 + l2**2 * jnp.cos(delta) + 2 * l1 * l2 * jnp.cos(delta))
            / (l2 * jnp.sin(delta))
        ) - C

    @partial(jit, static_argnums=(0,))
    def func_grad(self, delta, C):
        return grad(self.func, (0))(delta, C)

    @partial(jit, static_argnums=(0,))
    def pure_pursuit_controller_old_jax(self, X, ref_trajectory):
        # lookahead_distance = 5.0
        # Xd = compute_ref_point_from_trajectory( ref_trajectory )
        # alpha =

        Xd = self.compute_ref_point_from_trajectory(ref_trajectory)

        ld = jnp.linalg.norm(Xd[0:2] - X[0:2])
        yaw = X[2]
        k_alpha = 0.3
        alpha = k_alpha * (jnp.arctan2(Xd[1] - X[1], Xd[0] - X[0]) - yaw)

        alpha = lax.cond(
            jnp.abs(alpha) < 0.1, lambda x: 0.1 * jnp.sign(x), lambda x: x, alpha
        )
        C = ld / 2 / jnp.sin(alpha)

        # steering_desired = X[3] # initial guess
        steering_desired = 0.03 * jnp.sign(C)
        lr = 0.01

        # jax.debug.print("C = {x}, delta: {y}", x = C, y = X[3])

        @jit
        def body(i, inputs):
            steering_desired, C = inputs
            value = self.func(steering_desired, C)
            der = self.func_grad(steering_desired, C)
            der = lax.cond(
                jnp.abs(der) < 0.01, lambda x: 0.01 * jnp.sign(der), lambda x: x, der
            )
            # jax.debug.print("inside value = {x}", x = value)
            # jax.debug.print("st: {x}, der: {y}", x = steering_desired, y = der)
            return jnp.clip(
                steering_desired - lr * value / der, -jnp.pi / 3, jnp.pi / 3
            ), C  # + jnp.clip(-lr * value / der, -0.01, 0.01)

        steering_desired, _ = lax.fori_loop(0, 3000, body, (steering_desired, C))

        if self.pure_pursuit_mode == "front":
            steering_desired = -steering_desired

        # jax.debug.print("value = {x}", x = self.func(steering_desired, C))
        steering_rate = (steering_desired - X[3]) / self.dt
        steering_rate = jnp.clip(
            steering_rate, -self.steering_rate_bound, self.steering_rate_bound
        )

        speed_des = 3.0 * jnp.cos(alpha / 2)  # 3.0
        speed = X[4]
        acc = (speed_des - speed) / self.dt
        acc = jnp.clip(acc, -self.acc_bound, self.acc_bound)
        U = jnp.array([steering_rate, acc])

        # jax.debug.print(
        #     "value: {value}, ld = {x}, alpha = {y}, ref: {z}, yaw: {w}, steering: {uu}, pos: {pos}, input: {U}",
        #     value=self.func(steering_desired, C),
        #     x=ld,
        #     y=alpha,
        #     z=Xd,
        #     w=yaw,
        #     uu=steering_desired,
        #     pos=X[0:2],
        #     U=U,
        # )

        return U, Xd

    @partial(jit, static_argnums=(0,))
    def pure_pursuit_controller_jax(self, X, ref_trajectory, cur_index):
        # Xd, cur_ref_index = self.compute_closest_ref_point_from_trajectory(ref_trajectory, cur_index, X)
        Xd, cur_ref_index = self.compute_ref_point_from_trajectory(
            ref_trajectory, cur_index, X
        )

        ld = jnp.linalg.norm(Xd[0:2] - X[0:2])
        yaw = X[2]
        k_alpha = 1.0
        alpha = k_alpha * (jnp.arctan2(Xd[1] - X[1], Xd[0] - X[0]) - yaw)

        alpha = lax.cond(
            jnp.abs(alpha) < 0.01, lambda x: 0.01 * jnp.sign(x), lambda x: x, alpha
        )
        R = ld / 2 / jnp.sin(alpha)

        steering_desired = (
            jnp.pi
            - jnp.arctan(jnp.abs(R) / self.hitch_to_dump_bed_rear_axle)
            - jnp.arccos(
                self.hitch_to_cabin_axle
                / jnp.sqrt(
                    R * R
                    + self.hitch_to_dump_bed_rear_axle
                    * self.hitch_to_dump_bed_rear_axle
                )
            )
        )
        steering_desired = steering_desired * jnp.sign(R)
        steering_desired = jnp.clip(
            steering_desired, -self.steering_bound, self.steering_bound
        )

        if self.pure_pursuit_mode == "front":
            steering_desired = -steering_desired

        # jax.debug.print("value = {x}", x = self.func(steering_desired, C))
        steering_rate = (steering_desired - X[3]) / self.dt
        steering_rate = jnp.clip(
            steering_rate, -self.steering_rate_bound, self.steering_rate_bound
        )

        speed_des = 3.0  # * jnp.cos(alpha / 2)  # 3.0
        speed = X[4]
        acc = (speed_des - speed) / self.dt
        acc = jnp.clip(acc, -self.acc_bound, self.acc_bound)
        U = jnp.array([steering_rate, acc])

        # jax.debug.print(
        #     # "value: {value},
        #     "ld = {x}, alpha = {y}, ref: {z}, yaw: {w}, steering: {delta}, des steering: {uu}, pos: {pos}, input: {U}, R: {r}",
        #     # value=self.func(steering_desired, C),
        #     x=ld,
        #     y=alpha,
        #     z=Xd,
        #     w=yaw,
        #     delta=X[3],
        #     uu=steering_desired,
        #     pos=X[0:2],
        #     U=U,
        #     r=R,
        # )

        return U, cur_ref_index

    @partial(jit, static_argnums=(0,))
    def compute_ref_point_from_trajectory(self, ref_trajectory, cur_index, cur_pos):
        cur_index += 1
        cur_index = jnp.clip(cur_index, 0, ref_trajectory.shape[1] - 1)
        return ref_trajectory[:, cur_index], cur_index

    @partial(jit, static_argnums=(0,))
    def compute_closest_ref_point_from_trajectory(
        self, ref_trajectory, cur_index, cur_pos
    ):
        def cond(state):
            val, count = state
            return jnp.logical_and(
                val < self.lookahead_distance, count < ref_trajectory.shape[1]
            )

        def body(state):
            val, count = state
            count += 1
            val = jnp.linalg.norm(ref_trajectory[:, count] - cur_pos[0:2])
            jax.debug.print("dist: {dist}, ind: {ind}", dist=val, ind=count)
            return (val, count)

        dist, cur_index = lax.while_loop(cond, body, (0, cur_index))

        return ref_trajectory[:, cur_index - 1], cur_index - 1

    @partial(jit, static_argnums=(0,))
    def marco_controller_jax(self, X, ref_trajectory):
        Xd = self.compute_ref_point_from_trajectory(ref_trajectory)

        x = X[0]
        y = X[1]
        psi = X[2]
        steering = X[3]
        V = X[4]

        l1 = self.hitch_to_dump_bed_rear_axle
        l2 = self.hitch_to_cabin_axle

        # Polar coordinate states
        r = jnp.linalg.norm(X[0:2] - Xd)
        phi = jnp.arctan2(Xd[1] - y, Xd[0] - x)
        delta = psi - phi

        # Slow loop - Lyapunov stable
        k_phi = 1.0
        delta_desired = jnp.arctan(-k_phi * phi)

        # Fast loop
        k_delta = 0.3
        curvature = (
            -1
            / r
            * (
                k_delta * (delta - delta_desired)
                + (1 + k_phi / (1 + jnp.square(k_phi * phi))) * jnp.sin(delta)
            )
        )
        v_max = 2.0
        lambdaa = 1.0
        beta = 0.3
        curvature_max = 0.1136
        v_op = v_max * (1 - beta * jnp.pow(jnp.abs(curvature_max), lambdaa))
        v = v_op / (1 - beta * jnp.pow(jnp.abs(curvature), lambdaa))

        # omega = curvature * v
        L = l2 + l1 * jnp.cos(steering)
        curvature_star = -(L / l2 * curvature + jnp.sin(steering) / l2)
        steering_rate = curvature_star * v

        long_acc = (v - V) / self.dt
        return jnp.array([steering_rate, long_acc])

    @partial(jit, static_argnums=(0,))
    def cabin_state_from_dump_bed_state(self, X, U):
        x = X[0]
        y = X[1]
        psi = X[2]
        steering = X[3]
        V = X[4]
        steering_rate = U[0]
        l1 = self.hitch_to_dump_bed_rear_axle
        l2 = self.hitch_to_cabin_axle
        psi_cabin = psi + steering
        x_cabin = x + l1 * jnp.cos(psi) + l2 * jnp.cos(psi_cabin)
        y_cabin = y + l1 * jnp.sin(psi) + l2 * jnp.sin(psi_cabin)
        v_cabin = V * jnp.cos(
            steering
        ) + steering_rate * self.hitch_to_dump_bed_rear_axle * jnp.sin(steering)
        return jnp.array([x_cabin, y_cabin, psi_cabin, steering, v_cabin])

    @partial(jit, static_argnums=(0,))
    def dump_bed_state_from_cabin_state(self, X, U):
        x = X[0]
        y = X[1]
        psi = X[2]
        steering = X[3]
        V = X[4]
        steering_rate = U[0]
        l1 = self.hitch_to_dump_bed_rear_axle
        l2 = self.hitch_to_cabin_axle
        psi_dump_bed = psi - steering
        x_dump_bed = x - l1 * jnp.cos(psi_dump_bed) - l2 * jnp.cos(psi)
        y_dump_bed = y - l1 * jnp.sin(psi_dump_bed) - l2 * jnp.sin(psi)
        v_dump_bed = V * jnp.cos(
            steering
        ) + steering_rate * self.hitch_to_dump_bed_rear_axle * jnp.sin(steering)
        return jnp.array([x_dump_bed, y_dump_bed, psi_dump_bed, steering, v_dump_bed])

    @partial(jit, static_argnums=(0,))
    def frenet_controller_jax(self, X):
        U = np.array([0.0, 0.0])
        return U

    @partial(jit, static_argnums=(0,))
    def mpc_controller_jax(self, X):
        U = np.array([0.0, 0.0])
        return U


def animate_trajectory(ref=None, traj=None, traj_ref=None):
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

    first_frame = (
        hm400.render(traj[:, 0], traj_ref[:, 0])
        if traj_ref is not None
        else hm400.render(traj[:, 0])
    )
    for trace in first_frame:
        fig.add_trace(trace)

    frames = []
    for i in range(traj.shape[1] - 1):
        vehicle_traces = (
            hm400.render(traj[:, i], traj_ref[:, i])
            if traj_ref is not None
            else hm400.render(traj[:, i])
        )
        if ref is not None:
            frames.append(
                go.Frame(
                    data=[ref_trace, vehicle_path_trace] + vehicle_traces, name=str(i)
                )
            )
        else:
            frames.append(
                go.Frame(data=[vehicle_path_trace] + vehicle_traces, name=str(i))
            )

    # Add empty traces for vehicle parts to be updated by animation
    num_vehicle_animation_traces = (
        len(frames[0].data) if traj_ref is not None else len(frames[0].data) - 1
    )
    for _ in range(num_vehicle_animation_traces):
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
    if traj_ref is not None:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                line=dict(color="black"),
                name="Chosen reference",
            )
        )

    fig.update(frames=frames)

    fig.update_layout(
        xaxis=dict(range=[-20, 45]),
        yaxis=dict(range=[-20, 25]),
        width=2500,  # Set the width in pixels
        height=1500,  # Set the height in pixels
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
        ],  #plotly_update_menus,
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
    X = np.array([0, 0, 0, 0.1, 0])
    dt = 0.02  # 0.05
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


def main_controller():
    X = np.array([1.0, 1.0, 0, 0.1, 0])

    dt = 0.05
    tf = 60  # 2 * 60
    num_steps = int(tf / dt)
    times = np.linspace(0, tf, num_steps)
    speed = -3

    # radius = 30
    # Xref = np.vstack(
    #     (
    #         radius * np.cos(speed / radius * times),
    #         radius * np.sin(speed / radius * times),
    #     )
    # )

    Xref = np.vstack((10 + speed * times, np.ones(times.size) * 15))

    vehicle = HM400(dt=dt)
    # U = vehicle.marco_controller_jax(X, Xref[:, 0])
    U = vehicle.pure_pursuit_controller_jax(X, Xref, 0)
    # print(U)
    # exit()
    tf = times[-1]
    t = 0

    # Marco
    # traj = vehicle.dump_bed_state_from_cabin_state(X, jnp.array([0, 0])).reshape(-1, 1)
    # for iter in range(times.size):
    #     action = vehicle.marco_controller_jax(X, Xref[:, iter])
    #     X = vehicle.step_front_jax(X, action, dt)
    #     traj = np.append(
    #         traj,
    #         vehicle.dump_bed_state_from_cabin_state(X, jnp.array([0, 0])).reshape(
    #             -1, 1
    #         ),
    #         axis=1,
    #     )
    #     t += dt
    #     print(iter)
    # animate_trajectory(traj=traj, ref=Xref)

    # pure_pursuit_mode = "front"
    pure_pursuit_mode = "rear"
    vehicle = HM400(dt=dt, pure_pursuit_mode=pure_pursuit_mode)

    # Pure Pursuit front
    if pure_pursuit_mode == "front":
        print(f"FRONT")
        traj = vehicle.dump_bed_state_from_cabin_state(X, jnp.array([0, 0])).reshape(
            -1, 1
        )
        traj_ref = []
        cur_ref_index = 0
        for iter in range(times.size):
            action, cur_ref_index = vehicle.pure_pursuit_controller_jax(
                X, Xref, cur_ref_index
            )
            traj_ref = (
                np.append(traj_ref, Xref[:, cur_ref_index].reshape(-1, 1), axis=1)
                if iter > 0
                else Xref[:, cur_ref_index].reshape(-1, 1)
            )
            X = vehicle.step_front_jax(X, action, dt)
            traj = np.append(
                traj,
                np.asarray(
                    vehicle.dump_bed_state_from_cabin_state(
                        X, jnp.array([0, 0])
                    ).reshape(-1, 1)
                ),
                axis=1,
            )
            t += dt
            # print(iter)
        animate_trajectory(
            traj=np.asarray(traj), ref=np.asarray(Xref), traj_ref=traj_ref
        )

    # Pure Pursuit rear
    if pure_pursuit_mode == "rear":
        print(f"REAR")
        traj = np.asarray(X).reshape(-1, 1)
        traj_ref = []
        cur_ref_index = 0
        for iter in range(times.size):
            action, cur_ref_index = vehicle.pure_pursuit_controller_jax(
                X, Xref, cur_ref_index
            )
            traj_ref = (
                np.append(traj_ref, Xref[:, cur_ref_index].reshape(-1, 1), axis=1)
                if iter > 0
                else Xref[:, cur_ref_index].reshape(-1, 1)
            )
            X = vehicle.step_rear_jax(X, action, dt)
            traj = np.append(traj, np.asarray(X.reshape(-1, 1)), axis=1)
            t += dt
            # print(iter)
        # exit()
        animate_trajectory(
            traj=np.asarray(traj), ref=np.asarray(Xref), traj_ref=traj_ref
        )

    # traj = X.reshape(-1, 1)
    # for iter in range(times.size):
    #     # action = vehicle.pure_pursuit_controller_jax(X, Xref[:,iter])
    #     if t < tf / 2:
    #         action = np.array([0.1, 0.4])
    #     else:
    #         action = np.array([-0.1, 0.4])
    #     X = vehicle.step_rear_jax(X, action, dt)
    #     traj = np.append(traj, np.asarray(X.reshape(-1, 1)), axis=1)
    #     t += dt
    #     print(iter)
    # animate_trajectory(traj=np.asarray(traj), ref = np.asarray(Xref))

    # traj = vehicle.dump_bed_state_from_cabin_state(X, jnp.array([0,0])).reshape(-1, 1)
    # for iter in range(times.size):
    #     # action = vehicle.pure_pursuit_controller_jax(X, Xref[:,iter])
    #     if t < tf / 2:
    #         action = np.array([0.1, 0.4])
    #     else:
    #         action = np.array([-0.1, 0.4])
    #     X = vehicle.step_rear_jax(X, action, dt)
    #     traj = np.append(traj, np.asarray(vehicle.dump_bed_state_from_cabin_state(X, jnp.array([0,0])).reshape(-1, 1)), axis=1)
    #     t += dt
    #     print(iter)
    # animate_trajectory(traj=np.asarray(traj), ref = np.asarray(Xref))


if __name__ == "__main__":
    main_controller()

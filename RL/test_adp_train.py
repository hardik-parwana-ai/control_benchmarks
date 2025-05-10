from __future__ import annotations

from dataclasses import dataclass
import pdb
import socket
import typing
from typing import Any, Callable
import unittest

import gymnasium as gym

# from adp.lib.rl_sim.adp_minimal import service_starter
from adp.lib.rl_sim.lib import adp_gym
from adp.lib.rl_sim.lib import adp_types
from adp.lib.rl_sim.lib import data_api
from adp.lib.rl_sim.lib import environment
from adp.lib.rl_sim.lib import space_factories
from adp.lib.rl_sim.logging import logger as logger_module
from simian.public.proto import spatial_pb2
from simian.public.proto.map import map_pb2


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port.
        port = s.getsockname()[1]
    return int(port)


@dataclass
class Observation:
    traffic_light_states: adp_types.TrafficLightStates
    map_data: map_pb2.Map
    next_sim_time_ns: int
    ego_pose: spatial_pb2.Pose
    ego_velocity: spatial_pb2.Screw
    ego_acceleration: spatial_pb2.Screw


class FakeObservationModel(environment.ObservationModel[Observation]):
    @property
    def observation_space(self) -> gym.spaces.Space[typing.Any]:
        return None  # type: ignore

    def observe(self, data_api: data_api.DataAPI) -> Observation:
        # print(data_api.get_local_map(origin=(100, 100), radius_meters=100))
        return Observation(
            traffic_light_states=data_api.get_traffic_light_states(),
            map_data=data_api.get_local_map(origin=(100, 100), radius_meters=100),
            next_sim_time_ns=data_api.get_timestamp_ns(),
            ego_pose=data_api.get_agent_pose(data_api.get_ego_id()),
            ego_velocity=data_api.get_agent_velocity(data_api.get_ego_id()),
            ego_acceleration=data_api.get_agent_acceleration(data_api.get_ego_id()),
        )


class FakeRewardModel(environment.RewardModel):
    def calculate_reward(self, data_api: data_api.DataAPI) -> environment.StepReward:
        _ = data_api
        return {"reward": 0.0, "info": {}}


class FakeTerminationModel(environment.TerminationModel):
    def calculate_termination(self, data_api: data_api.DataAPI) -> bool:
        _ = data_api
        return False


class BaseADPGymTest(unittest.TestCase):
    def __init__(
        self,
        no_sim_executor_setup: bool,
        adp_container_address: str,
        sim_executor_port: int,
        external_sim_control_port: int,
        scenario_name: str,
        workspace_dir: str,
        postconditions_callback: Callable[[Observation], None] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.no_sim_executor_setup = no_sim_executor_setup
        self.adp_container_address = adp_container_address
        self.sim_executor_port = sim_executor_port if sim_executor_port != 0 else _find_free_port()
        self.external_sim_control_port = (
            external_sim_control_port if external_sim_control_port != 0 else _find_free_port()
        )
        self.scenario_name = scenario_name
        self.workspace_dir = workspace_dir
        self.postconditions_callback = postconditions_callback

    def setUp(self) -> None:
        self.logger = logger_module.create_logger("adp_gym_test")

        # if not self.no_sim_executor_setup:
        #     self._services = service_starter.start_services(
        #         simulation_executor_service_port=self.sim_executor_port,
        #         logger=self.logger,
        #     )
        #     for service in self._services:
        #         self.addCleanup(service.process.terminate)

    def test_run_simulation_to_completion(self) -> None:
        self.logger.info("Preparing workspace: %s", self.workspace_dir)

        client = adp_gym.ADPClient(
            adp_container_address=self.adp_container_address,
            sim_executor_port=self.sim_executor_port,
            external_sim_control_port=self.external_sim_control_port,
            workspace_path=self.workspace_dir,
        )

        self.logger.info("Making the environment.")
        env = environment.ADPSimulatorEnv(
            FakeObservationModel(),
            FakeRewardModel(),
            FakeTerminationModel(),
            client,
            self.scenario_name,
        )

        self.logger.info("Calling .reset()")
        observation, _ = env.reset()

        step = 0
        terminated = False
        traffic_light_color = ["red", "green", "yellow"]

        while not terminated:
            self.logger.info("Calling .step() for step %d", step)
            env.set_next_traffic_light_states({step: traffic_light_color[step % 3]})
            observation, _, terminated, _, _ = env.step(
                space_factories.ADPAction(
                    agent_id=10,
                    agent_command=space_factories.NormalizedDBW(
                        brake=0.0,
                        throttle=1.0,
                        steering=0.0,
                    ),
                )
            )
            # print(f"observations: {observation.map_data.header}")
            # print(f"ego state: {observation.ego_pose.pz}")
            # print(f"ego velocity: {observation.ego_velocity}")
            # print(f"ego acceleration: {observation.ego_acceleration}")
            # print(f"traffic light states: {observation.traffic_light_states}")
            # print(f"map: {observation.map_data}")
            # map_pb = map_pb2()
            # map_pb.ParseFromString(observation.map_data)
            # with open("map.pb", "rb") as f:
            #     map_pb.ParseFromString(f.read())
            for lane in observation.map_data.lane:
                print("Lane ID:", lane.id.id)
                if lane.id.id == "_2":
                    print(f"lane data: {lane}")
                    # print(f"Lane central line data: {lane.central_curve.segment}")
                    # print(
                    #     f"Lane central line data: {lane.central_curve.segment[0].line_segment.point[0]}"
                    # )
                    # print(f"Lane left boundary: {}")
            # pdb.set_trace()
            exit()

            self.assertEqual(
                observation.traffic_light_states, {step: traffic_light_color[step % 3]}
            )
            if self.postconditions_callback:
                self.postconditions_callback(observation)

            step += 1

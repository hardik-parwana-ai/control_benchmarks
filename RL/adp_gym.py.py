from __future__ import annotations

import logging

from adp.lib.rl_sim.lib import simulation_executor_service_client
from adp.lib.rl_sim.simulation_executor_service.proto import simulation_executor_service_pb2

DEFAULT_SIM_EXECUTOR_PORT = 10010
DEFAULT_EXTERNAL_SIM_CONTROL_PORT = 10011

# print(f"hello")
class ADPClient:
    def __init__(
        self,
        workspace_path: str,
        adp_container_address: str,
        sim_executor_port: int = DEFAULT_SIM_EXECUTOR_PORT,
        external_sim_control_port: int = DEFAULT_EXTERNAL_SIM_CONTROL_PORT,
        logger: logging.Logger | None = None,
    ) -> None:
        self._workspace_path = workspace_path
        self._adp_container_address = adp_container_address
        self._sim_executor_port = sim_executor_port
        self._external_sim_control_port = external_sim_control_port
        self._logger = logger or logging.getLogger(__name__)

        self._sim_exec_service_client = (
            simulation_executor_service_client.SimulationExecutorServiceClient.FromServiceAddress(
                f"{self._adp_container_address}:{self._sim_executor_port}", self._logger
            )
        )

    def start_sim(
        self, scenario_path: str
    ) -> simulation_executor_service_pb2.StartSimulationResponse:
        return self._sim_exec_service_client.StartSimulation(
            scenario_path=scenario_path,
            workspace_path=self._workspace_path,
            agent_control_mode=simulation_executor_service_pb2.StartSimulationRequest.AgentControlMode.AGENT_CONTROL_MODE_EXTERNAL,
            external_sim_control_port=self._external_sim_control_port,
        )

    def is_sim_running(self) -> bool:
        return (
            self._sim_exec_service_client.PollSimulation()
            == simulation_executor_service_pb2.PollSimulationResponse.SimulationStatus.SIMULATION_STATUS_RUNNING
        )

    def terminate_sim(self) -> None:
        self._sim_exec_service_client.TerminateSimulation()

    def get_external_sim_control_address(self) -> str:
        return f"{self._adp_container_address}:{self._external_sim_control_port}"

from __future__ import annotations

from typing import Any
import unittest

from adp.lib.resources import resources
from adp.lib.rl_sim.integration_tests import adp_gym_test_lib


def postconditions_callback(obs: adp_gym_test_lib.Observation) -> None:
    # Our Ego has a heading "up" the x axis, and is constantly accelerating
    # according to our input above multiplied by its capabilities.
    assert len(obs.map_data.lane) == 8
    assert abs(obs.ego_acceleration.tx - 1.5) < 1e-6
    assert abs(obs.ego_acceleration.ty - 0.0) < 1e-6
    assert abs(obs.ego_acceleration.tz - 0.0) < 1e-6


class ADPGymTest(adp_gym_test_lib.BaseADPGymTest):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        no_sim_executor_setup = False
        adp_container_address = "localhost"
        sim_executor_port = 0  # pick one for me
        external_sim_control_port = 0  # pick one for me

        scenario_name = resources.GetResourceFilename(
            "adp/lib/rl_sim/testing/test_workspace/compiled_scenarios/smooth-lane-keeping-scenario-1742855419338.compiled"
        )
        workspace_dir = resources.GetResourceFilename("adp/lib/rl_sim/testing/test_workspace")

        # sanity check (data deps are hard)
        # try to read one byte from the scenario file to make sure it's actually there
        open(scenario_name, "rb").read(1)

        super().__init__(
            no_sim_executor_setup,
            adp_container_address,
            sim_executor_port,
            external_sim_control_port,
            scenario_name,
            workspace_dir,
            postconditions_callback,
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
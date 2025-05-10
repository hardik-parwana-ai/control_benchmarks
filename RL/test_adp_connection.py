
from adp.lib.rl_sim.lib import adp_gym

print(f"Starting client")
client = adp_gym.ADPClient(
            # adp_container_address="rl-sim-network",
            adp_container_address="applied_dev",
            sim_executor_port=10230,
            external_sim_control_port=10231,
            workspace_path="",
        )
print(f"client initialized")
import json
from free_range_zoo.envs import wildfire_v0
from free_range_zoo.envs.wildfire.configs.aaai_2024 import aaai_2025_ol_config
from free_range_zoo.wrappers.action_task import action_mapping_wrapper_v0
from free_range_zoo.envs.wildfire.env.utils.rendering import render
import torch
import pickle
import warnings
import os
import shutil

# print(torch.cuda.is_available())
# Suppress the specific nested tensor warning
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

with open('./WS1.pkl','rb') as f:
    wildfire_configuration = pickle.load(f)

# Add the missing termination_kappa attribute

wildfire_configuration.reward_config.termination_kappa = 1  # Set a reasonable default value

# print("-----------------------------------")
print(wildfire_configuration)
# print("-----------------------------------")
# count = 0
log_dir = "outputs/wildfire_logging_test_0"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)  # Remove the directory if it exists
# os.makedirs(log_dir)  # Create a fresh directory for logging

env = wildfire_v0.parallel_env(
    max_steps = 700,
    parallel_envs = 1,
    configuration = wildfire_configuration,
    device=torch.device('cpu'),
    log_directory = log_dir,
    override_initialization_check = True
)
env.reset()
env = action_mapping_wrapper_v0(env)

observations, infos = env.reset()
from free_range_zoo.envs.wildfire.baselines import NoopBaseline, RandomBaseline,StrongestBaseline,WeakestBaseline
# Check how many agents are available
# print("Number of agents:", len(env.agents))
# print("Available agents:", env.agents)

# Create agents
agents = {
    env.agents[0]:StrongestBaseline (agent_name = "firefighter_1", parallel_envs = 1),
    env.agents[1]: WeakestBaseline(agent_name = "firefighter_2", parallel_envs = 1),
    env.agents[2]: StrongestBaseline(agent_name = "firefighter_3", parallel_envs = 1),
    # env.agents[3]: RandomBaseline(agent_name = "firefighter_4", parallel_envs = 1),
}

while not torch.all(env.finished):
    for agent_name, agent in agents.items():
        agent.observe(observations[agent_name])  # Policy observation 
    agent_actions = {
            agent_name:agents[agent_name].act(action_space = env.action_space(agent_name))
        for agent_name in env.agents
    }  # Policy action determination here
    
    agent_actions = {
        k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        for k, v in agent_actions.items()
        
    }
    observations, rewards, terminations, truncations, infos = env.step(agent_actions)

    
env.close()
render("outputs/wildfire_logging_test_0/done.csv", render_mode="human", frame_rate=15)
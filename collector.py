from collections import OrderedDict
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Any

import gym
import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from agent import Agent
from utils import with_default_config, unpack, DataBatch, discount_rewards_to_go, discount_td_rewards

from unityagents import UnityEnvironment
from unityagents.brain import BrainInfo


class Memory:
    """
    Holds the rollout data in a dictionary
    """

    def __init__(self):
        """
        Creates the memory container. 
        """

        # Dictionaries to hold all the relevant data, with appropriate type annotations
        _observations: List[np.ndarray] = []
        _actions: List[int] = []
        _rewards: List[float] = []
        _logprobs: List[float] = []
        _entropies: List[float] = []
        _dones: List[bool] = []
        _values: List[float] = []

        self.data = {
            "observations": _observations,
            "actions": _actions,
            "rewards": _rewards,
            "logprobs": _logprobs,
            "entropies": _entropies,
            "values": _values,
            "dones": _dones,
        }

    def store(self,
              obs: np.ndarray,
              action: np.ndarray,
              reward: float,
              logprob: float,
              entropy: float,
              value: float,
              done: bool):

        update = (obs, action, reward, logprob, entropy, value, done)
        for key, var in zip(self.data, update):
            self.data[key].append(var)

    def reset(self):
        for key in self.data:
            self.data[key] = []

    def get_torch_data(self, gamma: float = 0.99, tau: float = 0.95) -> DataBatch:
        """
        Gather all the recorded data into torch tensors (still keeping the dictionary structure)
        """
        observations = torch.tensor(np.stack(self.data["observations"]))
        actions = torch.tensor(np.stack(self.data["actions"]))
        rewards = torch.tensor(self.data["rewards"])
        logprobs = torch.tensor(self.data["logprobs"])
        entropies = torch.tensor(self.data["entropies"])
        values = torch.tensor(self.data["values"])
        dones = torch.tensor(self.data["dones"])

        # rewards_to_go = discount_rewards_to_go(rewards, dones, gamma)
        #
        # advantages = (rewards_to_go - values)
        # advantages = (advantages - advantages.mean())
        # advantages = advantages / (torch.sqrt(torch.mean(advantages ** 2)) + 1e-8)

        torch_data = {
            "observations": observations,  # (batch_size, obs_size) float
            "actions": actions,  # (batch_size, ) int
            "rewards": rewards,  # (batch_size, ) float
            "logprobs": logprobs,  # (batch_size, ) float
            "entropies": entropies,
            "dones": dones,  # (batch_size, ) bool
            "values": values,
        }

        rewards_to_go, advantages = discount_td_rewards(torch_data, gamma, tau)

        advantages = (advantages - advantages.mean()) / advantages.std()

        torch_data = {
            "observations": observations[:-1],
            "actions": actions[:-1],
            "rewards": rewards[:-1],
            "logprobs": logprobs[:-1],
            "entropies": entropies[:-1],
            "dones": dones[:-1],
            "values": values[:-1],
            "rewards_to_go": rewards_to_go,
            "advantages": advantages
        }

        return torch_data

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


class Collector:
    """
    Class to perform data collection from two agents.
    """

    def __init__(self, agent: Agent, env: UnityEnvironment, brain_name: str = "ReacherBrain"):
        self.env = env
        self.agent = agent
        self.memory = Memory()
        self.brain_name = brain_name

    def collect_data(self,
                     num_steps: Optional[int] = None,
                     deterministic: bool = False,
                     disable_tqdm: bool = True,
                     train_mode: bool = True) -> DataBatch:
        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            deterministic: whether each agent should use the greedy policy; False by default
            disable_tqdm: whether a live progress bar should be (not) displayed
            train_mode:

        Returns: dictionary with the gathered data
        """

        self.reset()

        obs, _, _, _ = unpack(self.env.reset(train_mode=train_mode)[self.brain_name])

        for step in trange(num_steps+1, disable=disable_tqdm):
            # Compute the action for each agent
            action, logprob, entropy, value = self.agent.compute_single_action(obs, deterministic)

            # Actual step in the environment
            next_obs, reward, done, info = unpack(self.env.step(action)[self.brain_name])

            # Saving to memory
            self.memory.store(obs, action, reward, logprob, entropy, value, done)
            obs = next_obs

        return self.memory.get_torch_data()

    def reset(self):
        self.memory.reset()

    def change_agent(self, agent: Agent):
        """Replace the agent in the collector"""
        self.agent = agent

    def update_agent_state_dict(self, state_dict: Dict):
        """Update the state dict of the agent in the collector"""
        self.agent.model.load_state_dict(state_dict)


if __name__ == '__main__':
    pass

    # env = foraging_env_creator({})
    #
    # agent_ids = ["Agent0", "Agent1"]
    #
    # agents: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents, env, {})
    #
    # data_steps = runner.collect_data(num_steps=1000, disable_tqdm=False)
    # data_episodes = runner.collect_data(num_episodes=2, disable_tqdm=False)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')

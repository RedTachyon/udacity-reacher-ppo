import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agent import Agent
from unityagents import UnityEnvironment
from utils import Timer, with_default_config, write_dict
from collector import Collector
from policy_optimization import PPOptimizer


class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: UnityEnvironment,
                 config: Dict[str, Any]):
        self.agent = agent
        self.env = env
        self.config = config

    def train(self, num_iterations: int,
              disable_tqdm: bool = False,
              save_path: Optional[str] = None,
              **collect_kwargs):
        raise NotImplementedError


class PPOTrainer(Trainer):
    """This performs training in a sampling paradigm, where each agent is stored, and during data collection,
    some part of the dataset is collected with randomly sampled old agents"""
    def __init__(self, agent: Agent, env: UnityEnvironment, config: Dict[str, Any]):
        super().__init__(agent, env, config)

        default_config = {
            "steps": 2048,

            # Tensorboard settings
            "tensorboard_name": None,  # str, set explicitly

            # PPO
            "ppo_config": {
                # GD settings
                "optimizer": "adam",
                "optimizer_kwargs": {
                    "lr": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-7,
                    "weight_decay": 0,
                    "amsgrad": False
                },
                "gamma": .99,  # Discount factor

                # PPO settings
                "ppo_steps": 25,  # How many max. gradient updates in one iterations
                "eps": 0.1,  # PPO clip parameter
                "target_kl": 0.01,  # KL divergence limit
                "value_loss_coeff": 0.1,
                "entropy_coeff": 0.1,
                "max_grad_norm": 0.5,

                # Backpropagation settings
                "use_gpu": False,
            }
        }

        self.config = with_default_config(config, default_config)

        self.collector = Collector(agent=self.agent, env=self.env)
        self.ppo = PPOptimizer(agent=agent, config=self.config["ppo_config"])

        # Setup tensorboard
        self.writer: SummaryWriter
        if self.config["tensorboard_name"]:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path = Path.home() / "drlnd_logs" / f"{self.config['tensorboard_name']}_{dt_string}"
            self.writer = SummaryWriter(str(self.path))

            # Log the configs
            with open(str(self.path / "trainer_config.json"), "w") as f:
                json.dump(self.config, f)

            with open(str(self.path / f"agent_config.json"), "w") as f:
                json.dump(self.agent.model.config, f)

            self.path = str(self.path)
        else:
            self.writer = None

    def train(self, num_iterations: int,
              save_path: Optional[str] = None,
              disable_tqdm: bool = False,
              **collect_kwargs):

        print(f"Begin training, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        # Store the first agent
        # saved_agents = [copy.deepcopy(self.agent.model.state_dict())]

        if save_path:
            torch.save(self.agent.model, os.path.join(save_path, "base_agent.pt"))

        rewards = []

        for step in trange(num_iterations, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            # data_batch = self.collector.collect_data(num_episodes=self.config["episodes"])
            data_batch = self.collector.collect_data(num_steps=self.config["steps"])
            data_time = timer.checkpoint()
            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(data_batch, step, writer=self.writer)

            eval_batch = self.collector.collect_data(num_steps=1001)
            reward = eval_batch['rewards'].sum().item()
            rewards.append(reward)

            end_time = step_timer.checkpoint()

            if step % 500 == 0:
                print(f"Current reward: {reward:.3f}, Avg over last 100 iterations: {np.mean(rewards[-100:]):.3f}")

            # Save the agent to disk
            if save_path:
                torch.save(self.agent.model.state_dict(), os.path.join(save_path, f"weights_{step + 1}"))

            # Write training time metrics to tensorboard
            time_metrics = {
                "agent/time_data": data_time,
                "agent/time_total": end_time,
                "agent/eval_reward": reward
            }

            write_dict(time_metrics, step, self.writer)

        return rewards


if __name__ == '__main__':
    pass

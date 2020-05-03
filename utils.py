import time

from torch.utils.tensorboard import SummaryWriter
from unityagents.brain import BrainInfo
from typing import Tuple, Dict, Type, Callable, Any, Union, Optional

import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adamax import Adamax
from torch.optim.sgd import SGD


DataBatch = Dict[str, Any]


def with_default_config(config: Dict, default: Dict) -> Dict:
    """
    Adds keys from default to the config, if they don't exist there yet.
    Serves to ensure that all necessary keys are always present.
    Now also recursive.

    Args:
        config: config dictionary
        default: config dictionary with default values

    Returns:
        config with the defaults added
    """
    if config is None:
        config = {}
    else:
        config = config.copy()
    for key in default.keys():
        if isinstance(default[key], dict):
            config[key] = with_default_config(config.get(key), default.get(key))
        else:
            config.setdefault(key, default[key])
    return config


def unpack(env_info: BrainInfo) -> Tuple[np.ndarray, float, bool, Dict]:
    state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    done = env_info.local_done[0]
    info = {}
    return state.astype(np.float32), reward, done, info


def discount_rewards_to_go(rewards: Tensor, dones: Tensor, gamma: float = 1.) -> Tensor:
    """
    Computes the discounted rewards to go, handling episode endings. Nothing unusual.
    """
    dones = dones.to(torch.int32)  # Torch can't handle reversing boolean tensors
    current_reward = 0
    discounted_rewards = []
    for reward, done in zip(rewards.flip(0), dones.flip(0)):
        if done:
            current_reward = 0
        current_reward = reward + gamma * current_reward
        discounted_rewards.insert(0, current_reward)
    return torch.tensor(discounted_rewards)


def discount_td_rewards(data: DataBatch, gamma: float = 0.99, tau: float = 0.95) -> Tuple[Tensor, Tensor]:
    returns_batch = []
    advantages_batch = []
    returns = data['values'][-1]
    advantages = 0

    for i in reversed(range(len(data['dones']) - 1)):
        terminals = ~data['dones'][i]

        rewards = data['rewards'][i]
        value = data['values'][i]
        next_value = data['values'][i + 1]

        returns = rewards + gamma * terminals * returns  # v(s) = r + y*v(s+1)

        # calc. of discounted advantage= A(s,a) + y^1*A(s+1,a+1) + ...
        td_error = rewards + gamma * terminals * next_value.detach() - value.detach()  # td_err=q(s,a) - v(s)
        advantages = advantages * tau * gamma * terminals + td_error

        advantages_batch.insert(0, advantages)
        returns_batch.insert(0, returns)

    return torch.tensor(returns_batch), torch.tensor(advantages_batch)



def get_optimizer(opt_name: str) -> Type[Optimizer]:
    """Gets an optimizer by name"""
    optimizers = {
        "adam": Adam,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "adamax": Adamax,
        "sgd": SGD
    }

    if opt_name not in optimizers.keys():
        raise ValueError(f"Wrong optimizer: {opt_name} is not a valid optimizer name. ")

    return optimizers[opt_name]


def get_activation(act_name: str) -> Callable[[Tensor], Tensor]:
    """Gets an activation function by name"""
    activations = {
        "relu": F.relu,
        "relu6": F.relu6,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "softmax": F.softmax,
        "gelu": lambda x: x * F.sigmoid(1.702 * x)
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x: Tensor):
        return x * F.sigmoid(1.702 * x)


def get_activation_module(act_name: str) -> nn.Module:
    """Gets an activation module by name"""
    activations = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "gelu": GELU,
    }

    if act_name not in activations.keys():
        raise ValueError(f"Wrong activation: {act_name} is not a valid activation function name.")

    return activations[act_name]


def get_initializer(init_name: str) -> Callable[[Tensor], None]:
    """Gets an initializer by name"""
    initializers = {
        "kaiming_normal": nn.init.kaiming_normal_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "xavier_uniform": nn.init.xavier_uniform_,
        "zeros": nn.init.zeros_
    }

    if init_name not in initializers.keys():
        raise ValueError(f"Wrong initialization: {init_name} is not a valid initializer name.")

    return initializers[init_name]


class Timer:
    """
    Simple timer to get temporal metrics. Upon calling .checkpoint(), returns the time since the last call of
    """

    def __init__(self):
        self.start = time.time()

    def checkpoint(self) -> float:
        now = time.time()
        diff = now - self.start
        self.start = now
        return diff


def write_dict(metrics: Dict[str, Union[int, float]],
               step: int,
               writer: Optional[SummaryWriter] = None):
    """Writes a dictionary to a tensorboard SummaryWriter"""
    if writer is not None:
        writer: SummaryWriter
        for key, value in metrics.items():
            writer.add_scalar(tag=key, scalar_value=value, global_step=step)


def get_episode_lens(done_batch: Tensor) -> Tuple[int]:
    """
    Based on the recorded done values, returns the length of each episode in a batch.
    Args:
        done_batch: boolean tensor which values indicate terminal episodes

    Returns:
        tuple of episode lengths
    """
    episode_indices = done_batch.cpu().cumsum(dim=0)[:-1]
    episode_indices = torch.cat([torch.tensor([0]), episode_indices])  # [0, 0, 0, ..., 1, 1, ..., 2, ..., ...]

    ep_ids, ep_lens_tensor = torch.unique(episode_indices, return_counts=True)
    ep_lens = tuple(ep_lens_tensor.cpu().numpy())

    return ep_lens


def get_episode_rewards(batch: DataBatch) -> np.ndarray:
    """Computes the total reward in each episode in a data batch"""
    ep_lens = get_episode_lens(batch['dones'])

    ep_rewards = np.array([torch.sum(rewards) for rewards in torch.split(batch['rewards'], ep_lens)])

    return ep_rewards


def matrix_diag(diagonal: Tensor):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def minibatches(data: Dict[str, Tensor], batch_size: int, shuffle: bool = True) -> Tuple[Tensor, Dict[str, Tensor]]:
    batch_start = 0
    batch_end = batch_size
    data_size = len(data['dones'])

    if shuffle:
        indices = torch.randperm(data_size)
        data = {k: val[indices] for k, val in data.items()}
    else:
        indices = torch.arange(data_size)

    while batch_start < data_size:
        batch = {key: value[batch_start:batch_end] for key, value in data.items()}

        batch_start = batch_end
        batch_end = min(batch_start + batch_size, data_size)

        yield indices[batch_start:batch_end], batch


class Batcher:

    def __init__(self, batch_size, data):
        self.batch_size = batch_size  # 2
        self.data = data  # [array[0,1,2,...65]]
        self.num_entries = len(data[0])  # 66
        # print (self.batch_size) ; print (self.num_entries); print (self.data)
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
        # print (self.data)


def index_data(data: DataBatch, indices: Tensor) -> DataBatch:
    return {k: val[indices] for k, val in data.items()}

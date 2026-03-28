"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        dims = [state_dim] + list(hidden_dims) + [action_dim * chunk_size]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.pop()  # remove final ReLU (no activation on output)
        
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        criterion = nn.MSELoss()
        return criterion(action_chunk, self.model(state).unflatten(-1, (self.chunk_size, self.action_dim)))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.model(state).unflatten(-1, (self.chunk_size, self.action_dim))


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        dims = [state_dim + action_dim * chunk_size + 1] + list(hidden_dims) + [action_dim * chunk_size]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.pop()  # remove final ReLU (no activation on output)
        
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(action_chunk.shape[:-2])
        action_chunk_interpolated = tau[:, None, None] * action_chunk + (1 - tau[:, None, None]) * noise
        assert state.shape[:-1] == action_chunk.shape[:-2]
        assert state.shape[-1] == self.state_dim
        assert action_chunk.shape[-2:] == (self.chunk_size, self.action_dim)
        assert action_chunk_interpolated.shape == action_chunk.shape
        assert state.shape[:-1] == tau.shape

        action_chunk_interpolated_flattened = action_chunk_interpolated.flatten(-2, -1)

        feature_in = torch.cat([state, action_chunk_interpolated_flattened, tau[:, None]], dim=-1)
        criterion = nn.MSELoss()
        return criterion(self.model(feature_in).unflatten(-1, (self.chunk_size, self.action_dim)), action_chunk - noise)


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_chunk_sampled = torch.randn(state.shape[:-1] + (self.chunk_size * self.action_dim,))
        for i in range(num_steps):
            velocity = self.model(torch.cat([state, action_chunk_sampled, torch.ones(state.shape[:-1] + (1,)) * i / num_steps], dim=-1))
            action_chunk_sampled += velocity/num_steps
        
        return action_chunk_sampled.unflatten(-1, (self.chunk_size, self.action_dim))


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

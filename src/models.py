import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class SlateRewardPredictor(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_actions: np.ndarray,
        reward_type: str,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.slate_size = n_unique_actions.shape[0]
        self.n_unique_actions = n_unique_actions.max()

        self.fc1 = nn.Linear(
            dim_context + self.n_unique_actions * self.slate_size, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
    ):
        n_rounds = state.shape[0]
        action = action_onehot.reshape((n_rounds, -1))
        input = torch.cat((state, action), axis=1)

        x = F.relu(self.fc1(input))
        x = self.fc2(x)

        if self.reward_type == "binary":
            x = F.sigmoid(x)

        return x.squeeze()


class DiscreteSlateAbstractionRewardPredictor(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_latent_abstraction: int,
        reward_type: str,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.n_latent_abstraction = n_latent_abstraction

        self.fc1 = nn.Linear(dim_context + n_latent_abstraction, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        abstraction: torch.Tensor,
    ):
        input = torch.cat((state, abstraction), axis=1)
        x = F.relu(self.fc1(input))
        x = self.fc2(x)

        if self.reward_type == "binary":
            x = F.sigmoid(x)

        return x.squeeze()


class ContinuousSlateAbstractionRewardPredictor(nn.Module):
    def __init__(
        self,
        dim_context: int,
        dim_latent_abstraction: int,
        reward_type: str,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.reward_type = reward_type

        self.fc1 = nn.Linear(dim_context + dim_latent_abstraction, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        abstraction: torch.Tensor,
    ):
        input = torch.cat((state, abstraction), axis=1)

        x = F.relu(self.fc1(input))
        x = self.fc2(x)

        if self.reward_type == "binary":
            x = F.sigmoid(x)

        return x.squeeze()


class DiscreteSlateEncoder(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_actions: int,
        n_latent_abstraction: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.slate_size = n_unique_actions.shape[0]
        self.n_unique_actions = n_unique_actions.max()
        self.n_latent_abstraction = n_latent_abstraction

        self.fc1 = nn.Linear(
            dim_context + self.n_unique_actions * self.slate_size, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, n_latent_abstraction)

    def forward(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
    ):
        """Sample one-hot vector with Gumbel-softmax trick."""
        n_rounds = state.shape[0]
        action = action_onehot.reshape((n_rounds, -1))
        input = torch.cat((state, action), axis=1)

        x = F.relu(self.fc1(input))
        x = self.fc2(x)

        latent = F.gumbel_softmax(x, hard=True)
        log_prob = torch.log((F.gumbel_softmax(x) * latent).sum(axis=1))
        return latent, log_prob

    def calc_latent_prob(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
        latent: torch.Tensor,
    ):
        """Calc latent probability given sampled latent."""
        n_rounds = state.shape[0]
        action = action_onehot.reshape((n_rounds, -1))
        input = torch.cat((state, action), axis=1)

        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        prob = (F.gumbel_softmax(x) * latent).sum(axis=1)
        return prob


class ContinuousSlateEncoder(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_actions: int,
        dim_latent_abstraction: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.slate_size = n_unique_actions.shape[0]
        self.n_unique_actions = n_unique_actions.max()

        self.fc1 = nn.Linear(
            dim_context + self.n_unique_actions * self.slate_size, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, dim_latent_abstraction)  # mean
        self.fc3 = nn.Linear(hidden_dim, dim_latent_abstraction)  # std

    def forward(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
    ):
        """Sample latent representation with reparametrization trick."""
        n_rounds = state.shape[0]
        action = action_onehot.reshape((n_rounds, -1))
        input = torch.cat((state, action), axis=1)

        x = F.relu(self.fc1(input))
        mean = self.fc2(x)
        std = F.softplus(self.fc3(x))
        noise = torch.normal(torch.zeros_like(std), std)

        latent = mean + noise
        log_prob = (
            -torch.log(std * np.sqrt(2 * np.pi)) - torch.square(noise / std) / 2
        ).sum(axis=1)
        return latent, log_prob

    def calc_latent_prob(
        self,
        state: torch.Tensor,
        action_onehot: torch.Tensor,
        latent: torch.Tensor,
    ):
        """Calc latent probability given sampled latent."""
        n_rounds = state.shape[0]
        action = action_onehot.reshape((n_rounds, -1))
        input = torch.cat((state, action), axis=1)

        x = F.relu(self.fc1(input))
        mean = self.fc2(x)
        std = F.softplus(self.fc3(x))

        deviation = latent - mean
        prob = torch.exp(-torch.square(deviation / std) / 2) / (
            std * math.sqrt(2 * math.pi)
        )
        return prob


class DiscreteSlateDecoder(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_actions: np.ndarray,
        n_latent_abstraction: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.slate_size = n_unique_actions.shape[0]
        self.n_latent_abstraction = n_latent_abstraction

        self.fcs = []
        self.fc1 = nn.Linear(dim_context + n_latent_abstraction, hidden_dim)
        for l in range(self.slate_size):
            self.fcs.append(nn.Linear(hidden_dim, n_unique_actions[l]))

    def forward(
        self,
        state: torch.Tensor,
        abstraction: torch.Tensor,
        return_type: str = "log_prob",
    ):
        """Sample one-hot vector with Gumbel-softmax trick."""
        input = torch.cat((state, abstraction), axis=1)
        x = F.relu(self.fc1(input))

        outputs = []
        if return_type == "log_prob":
            for l in range(self.slate_size):
                outputs.append(torch.log(F.gumbel_softmax(self.fcs[l](x))))
        elif return_type == "prob":
            for l in range(self.slate_size):
                outputs.append(F.gumbel_softmax(self.fcs[l](x)))
        else:
            for l in range(self.slate_size):
                outputs.append(F.gumbel_softmax(self.fcs[l](x)), hard=True)

        return outputs


class ContinuousSlateDecoder(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_actions: np.ndarray,
        dim_latent_abstraction: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.slate_size = n_unique_actions.shape[0]

        self.fcs = []
        self.fc1 = nn.Linear(dim_context + dim_latent_abstraction, hidden_dim)
        for l in range(self.slate_size):
            self.fcs.append(nn.Linear(hidden_dim, n_unique_actions[l]))

    def forward(
        self,
        state: torch.Tensor,
        abstraction: torch.Tensor,
        return_type: str = "log_prob",
    ):
        input = torch.cat((state, abstraction), axis=1)

        outputs = []
        x = F.relu(self.fc1(input))

        if return_type == "log_prob":
            for l in range(self.slate_size):
                outputs.append(F.log_softmax(self.fcs[l](x)))
        elif return_type == "prob":
            for l in range(self.slate_size):
                outputs.append(F.softmax(self.fcs[l](x)))
        else:
            for l in range(self.slate_size):
                outputs.append(torch.multinomial(F.softmax(self.fcs[l](x))))

        return outputs

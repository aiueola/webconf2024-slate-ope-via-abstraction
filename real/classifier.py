import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(
        self,
        dim_context: int,
        n_unique_action: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_context, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_unique_action)

    def forward(
        self,
        state: torch.Tensor,
    ):
        """Sample one-hot vector with Gumbel-softmax trick."""
        x = F.relu(self.fc1(state))
        x = self.fc2(x)

        action = F.gumbel_softmax(x, hard=True)
        log_prob = torch.log((F.gumbel_softmax(x) * action).sum(axis=1))
        return action, log_prob

    def logit(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

    def greedy(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.max(x, dim=1, keepdim=True)[0] == x

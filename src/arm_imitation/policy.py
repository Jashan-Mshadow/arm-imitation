import numpy as np
import torch
import torch.nn as nn


def obs(q, target):
    return np.concatenate([q, target]).astype(np.float32)


class Policy(nn.Module):
    """MLP: (3 joint angles, 2D target) -> 3 joint-velocity commands."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act(self, q, target):
        return self(torch.from_numpy(obs(q, target))).numpy()


def train(policy, X, Y, epochs=300, lr=1e-3, batch=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = nn.functional.mse_loss(policy(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())

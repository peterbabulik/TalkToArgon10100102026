# requirements: torch, triton
# timeout: 600
"""
================================================================================
ArgonOS v98 Swarm Heavy Computation Workload
================================================================================
Directives:
  # requirements: torch, triton
  # timeout: 600
================================================================================
"""

import sys
import torch
import torch.nn as nn


class SwarmDeepModel(nn.Module):

    def __init__(self, in_features: int = 128, out_features: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def execute_heavy_workload():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Swarm Workload] Running on device: {device}")
    model = SwarmDeepModel().to(device)
    x = torch.randn(64, 128, device=device)
    out = model(x)
    print(f"[Swarm Workload] Forward pass completed. Shape: {out.shape}")


if __name__ == "__main__":
    execute_heavy_workload()
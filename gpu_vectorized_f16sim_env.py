import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import gym
import time
from pathlib import Path
from nlplant import nlplant_

class F16SimDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def compute_extended_state(self, s):
        nlplant_()

    def forward(self, t, s):
        es = self.compute_extended_state(s)
        return es

class GPUVectorizedF16SimEnv:
    def __init__(self, preset_name, n, dt=0.01, solver="euler", device="cuda:0"):
        pass

    def obs(self):
        return self.s

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def reset(self):
        pass

    def step(self, u):
        pass

    def render(self, **kwargs):
        """Save rollout data to emulate rendering."""
        pass
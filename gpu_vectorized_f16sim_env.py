import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import gym
import time
from pathlib import Path
from nlplant import nlplant_

INIT_U = [14.3842921301, 0.0, 999.240528869, 0.0, 0.0680626236787, 0.0, 100.08096494, 0.121545455798, 
            0.0, 0.0, -0.031583522788, 0.0, 20000.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class F16SimDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def compute_extended_state(self, u):
        return nlplant_(u)

    def forward(self, t, u):
        es = self.compute_extended_state(u)
        return es

class GPUVectorizedF16SimEnv:
    def __init__(self, preset_name, n, dt=0.01, solver="euler", device="cuda:0"):
        self.num_states = 18
        self.num_actions = 18
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,))
        self.state_space = self.observation_space

        self.preset_name = preset_name
        self.n = n
        self.dt = dt
        self.solver = solver
        self.device = torch.device(device)
        self.dynamics = None
        self.step_count = 0
        self.saved_data = []

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
        self.s = torch.zeros((self.n, 18), device=self.device)
        self.u = torch.tensor(INIT_U, device=self.device)
        self.dynamics = F16SimDynamics()
        self.es = self.dynamics.compute_extended_state(self.u)
        self.step_count = 0
        self.saved_data = []
        return self.obs()

    def step(self, u):
        self.dynamics = F16SimDynamics()
        self.s = odeint(self.dynamics, self.s, torch.tensor([0., self.dt], device=self.device), method=self.solver)[1, :, :]
        self.es = self.dynamics.compute_extended_state(self.u)
        self.step_count += 1
        obs, reward, done, info = self.obs(), self.reward(), self.done(), self.info()
        if torch.all(done) and self.saved_data:
            filename = time.strftime("%Y%m%d-%H%M%S") + ".pth"
            Path("data").mkdir(parents=True, exist_ok=True)
            torch.save(self.saved_data, os.path.join("data", filename))
        return obs, reward, done, info

    def render(self, **kwargs):
        """Save rollout data to emulate rendering."""
        self.saved_data.append((self.s[0, :].cpu(), self.u[0, :].cpu(), self.es[0, :].cpu()))

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import time

    # Compare solvers
    env_euler = GPUVectorizedF16SimEnv("racecar", 1, solver="euler")
    env_rk = GPUVectorizedF16SimEnv("racecar", 1, solver="dopri5")
    env_rk8 = GPUVectorizedF16SimEnv("racecar", 1, solver="dopri8")

    traj_euler = [env_euler.reset().cpu().numpy()]
    traj_rk = [env_rk.reset().cpu().numpy()]
    traj_rk8 = [env_rk8.reset().cpu().numpy()]

    for i in range(600):
        u = INIT_U
        s_euler, _, _, _ = env_euler.step(torch.tensor(u, device=torch.device("cuda:0")))
        s_rk, _, _, _ = env_rk.step(torch.tensor(u, device=torch.device("cuda:0")))
        s_rk8, _, _, _ = env_rk8.step(torch.tensor(u, device=torch.device("cuda:0")))

        traj_euler.append(s_euler.cpu().numpy())
        traj_rk.append(s_rk.cpu().numpy())
        traj_rk8.append(s_rk8.cpu().numpy())

    plt.figure(dpi=300)
    plt.plot([s[0][0] for s in traj_euler], [s[0][1] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    plt.plot([s[0][0] for s in traj_rk8], [s[0][1] for s in traj_rk8], label="RK8")
    plt.legend()
    plt.axis("equal")

    # # Test large-scale parallelization
    # ns = [10 ** i for i in range(1)]
    # def measure_time(n, solver):
    #     env = GPUVectorizedF16SimEnv("racecar", n, solver=solver)
    #     u = torch.tensor([[0.1, 10.] for _ in range(n)], device=torch.device("cuda:0"))
    #     start_time = time.time()
    #     for i in tqdm(range(1000)):
    #         env.step(u)
    #     elapsed = time.time() - start_time
    #     return elapsed
    # times_euler = [measure_time(n, "euler") for n in ns]
    # times_rk = [measure_time(n, "dopri5") for n in ns]
    # times_rk8 = [measure_time(n, "dopri5") for n in ns]
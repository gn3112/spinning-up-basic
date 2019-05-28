from env_reacher_v2 import environment
import torch


class Env():
  def __init__(self):
    self._env = environment()

  def reset(self):
    self._env.reset_robot_position(random_=True)
    self._env.reset_target_position(random_=False)
    state = self._env.get_obs()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)

  def step(self, action):
    reward, done = self._env.step_(action[0].detach().numpy())
    state = self._env.get_obs()
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, done

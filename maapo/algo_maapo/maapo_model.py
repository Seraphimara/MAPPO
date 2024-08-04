import torch.nn as nn
import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from .actor import Actor
from ray.rllib.policy.view_requirement import ViewRequirement
from gym.spaces import Box
from .attention_critic import AttentionCritic
from .critic import Critic


class MAAPOModel(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        dim_obs = np.product(obs_space.shape)
        dim_action = np.product(action_space.shape)
        self.tau = model_config["custom_model_config"]["tau"]
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.actor = Actor(dim_obs, num_outputs, model_config)
        self.ego_critic = Critic(dim_obs, model_config)
        self.region_critic = AttentionCritic(dim_obs, dim_action)
        self.target_ego_critic = Critic(dim_obs, model_config)
        self.target_ego_critic.load_state_dict(self.ego_critic.state_dict())
        self.target_region_critic = AttentionCritic(dim_obs, dim_action)
        self.target_region_critic.load_state_dict(self.region_critic.state_dict())

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs.reshape(obs.shape[0], -1)
        logits = self.actor(obs)
        return logits, state

    def value_function(self):
        raise ValueError(
            "Value Function should not be called directly! "
            "Call ego_value_function or region_value_function instead!"
        )

    def ego_value_function(self, batch):
        obs = batch["obs"]
        return self.ego_critic(obs)

    def region_value_function(self, batch):
        region_obs_act = batch["region_obs_act"]
        return self.region_critic(region_obs_act)

    def target_ego_value_function(self, batch):
        obs = batch["obs"]
        return self.target_ego_critic(obs)

    def target_region_value_function(self, batch):
        region_obs_act = batch["region_obs_act"]
        return self.target_region_critic(region_obs_act)

    def all_value_function(self, batch):
        return self.ego_value_function(batch) + self.region_value_function(batch)

    def actor_variables(self):
        return list(self.actor.parameters())

    
    def ego_critic_variables(self):
        return list(self.ego_critic.parameters())

    def region_critic_variables(self):
        return list(self.region_critic.parameters())

    def ego_soft_update(self):
        for param_target, param in zip(self.target_ego_critic.parameters(), self.ego_critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def region_soft_update(self):
        for param_target, param in zip(self.target_region_critic.parameters(), self.region_critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



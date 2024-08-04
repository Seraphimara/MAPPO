import logging

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class Critic(nn.Module):
    def __init__(self, dim_obs, model_config: ModelConfigDict):
        nn.Module.__init__(self)
        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        prev_vf_layer_size = dim_obs
        assert prev_vf_layer_size > 0
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0),
                )
            )
            prev_vf_layer_size = size
        self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_vf_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        self.critic = nn.Sequential(self._value_branch_separate, self._value_branch)

    def forward(self, obs):
        return torch.reshape(self.critic(obs), [-1])

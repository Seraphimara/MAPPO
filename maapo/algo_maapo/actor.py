import logging

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, dim_obs, num_outputs, model_config: ModelConfigDict):
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")

        layers = []
        prev_layer_size = dim_obs
        self._logits = None

        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        self._logits = SlimFC(
                in_size=prev_layer_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
        )

        # Layer to add the log std vars to the state-dependent means.

        self._hidden_layers = nn.Sequential(*layers)
        self.actor = nn.Sequential(self._hidden_layers, self._logits)

    def forward(self, obs):
        logits = self.actor(obs)
        return logits

import torch
import torch.nn as nn
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    compute_gae_for_sample_batch,
)
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.numpy import convert_to_numpy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.utils.typing import ModelGradients

def compute_ego_advantage(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout["target_ego_values"], np.array([last_r])])
    delta_t = (rollout["ego_rewards"] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout["ego_advantages"] = discount_cumsum(delta_t, gamma * lambda_)
    rollout["ego_target"] = (rollout["ego_advantages"] + rollout["target_ego_values"]).astype(np.float32)
    rollout["ego_advantages"] = rollout["ego_advantages"].astype(np.float32)
    return rollout


def compute_region_advantage(rollout: SampleBatch, last_r: float, gamma: float = 1.0, lambda_: float = 1.0):
    vpred_t = np.concatenate([rollout["target_region_values"], np.array([last_r])])
    delta_t = (rollout["region_rewards"] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout["region_advantages"] = discount_cumsum(delta_t, gamma * lambda_)
    rollout["region_target"] = (rollout["region_advantages"] + rollout["target_region_values"]).astype(np.float32)
    rollout["region_advantages"] = rollout["region_advantages"].astype(np.float32)
    return rollout


class MAAPOPolicy(
    ValueNetworkMixin,
    LearningRateSchedule,
    EntropyCoeffSchedule,
    KLCoeffMixin,
    TorchPolicyV2,
):
    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)
        self.target_model_update_freq = 4
        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return {}

    def loss(self, model, dist_class, train_batch):
        logits, state_info = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = torch.mean(action_kl)
            # for this update
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = torch.mean(curr_entropy)
        # advantages = train_batch["ego_advantages"] + train_batch["region_advantages"]
        # advantages = train_batch["ego_advantages"]
        advantages = train_batch["region_advantages"]
        # advantages = train_batch[Postprocessing.ADVANTAGES]
        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_surrogate_loss = torch.mean(-surrogate_loss)

        # Compute a value function loss.
        assert self.config["use_critic"]
        if self.config["old_value_loss"]:

            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss1 = torch.pow(current_vf - value_target, 2.0)
                vf_clipped = prev_vf + torch.clamp(
                    current_vf - prev_vf, -self.config["vf_clip_param"], self.config["vf_clip_param"]
                )
                vf_loss2 = torch.pow(vf_clipped - value_target, 2.0)
                vf_loss = torch.max(vf_loss1, vf_loss2)
                return vf_loss

        else:

            def _compute_value_loss(current_vf, prev_vf, value_target):
                vf_loss = torch.pow(current_vf - value_target, 2.0)
                vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
                return vf_loss_clipped
        vf_loss = _compute_value_loss(
            current_vf=model.all_value_function(train_batch),
            prev_vf= train_batch[SampleBatch.VF_PREDS],
            value_target=train_batch[Postprocessing.VALUE_TARGETS]
        )
        mean_vf_loss = torch.mean(vf_loss)

        ego_vf_loss = _compute_value_loss(
            current_vf=model.ego_value_function(train_batch),
            prev_vf=train_batch["ego_values"],
            value_target=train_batch["ego_target"]
        )
        mean_ego_vf_loss = torch.mean(ego_vf_loss)

        region_vf_loss = _compute_value_loss(
            current_vf=model.region_value_function(train_batch),
            prev_vf=train_batch["region_values"],
            value_target=train_batch["region_target"]
        )
        mean_region_vf_loss = torch.mean(region_vf_loss)

        total_loss = torch.mean(
            -surrogate_loss
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (already processed through `torch.mean`),
        # if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_surrogate_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["mean_ego_vf_loss"] = mean_ego_vf_loss
        model.tower_stats["mean_region_vf_loss"] = mean_region_vf_loss
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss
        return [mean_ego_vf_loss, mean_region_vf_loss, total_loss]

    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "mean_ego_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_ego_vf_loss"))
                ),
                "mean_region_vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_region_vf_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
            }
        )

    def region_process(self, sample_batch, other_agent_batches):
        region_obs_act = []
        region_reward = []
        for index in range(sample_batch.count):
            region_obs_act_step = []
            region_reward_step = []
            ego_obs_act = np.concatenate((sample_batch["obs"][index], sample_batch["actions"][index]))
            region_obs_act_step.append(ego_obs_act)
            ego_reward = sample_batch["rewards"][index]
            region_reward_step.append(ego_reward)
            environmental_time_step = sample_batch["t"][index]
            neighbours = sample_batch['infos'][index]["neighbours"]
            neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]
            obs_list = []
            act_list = []
            for nei_count, (nei_name, nei_dist) in enumerate(zip(neighbours, neighbours_distance)):
                # if nei_dist > self.config["nei_distance"]:
                #     continue
                if nei_count >= self.config["num_region_agents"]:
                    break

                nei_act = None
                nei_obs = None
                nei_reward = None
                if nei_name in other_agent_batches:
                    _, nei_batch = other_agent_batches[nei_name]

                    match_its_step = np.where(nei_batch["t"] == environmental_time_step)[0]

                    if len(match_its_step) == 0:
                        pass
                    elif len(match_its_step) > 1:
                        raise ValueError()
                    else:
                        new_index = match_its_step[0]
                        nei_obs = nei_batch[SampleBatch.CUR_OBS][new_index]
                        nei_act = nei_batch[SampleBatch.ACTIONS][new_index]
                        nei_reward = nei_batch[SampleBatch.REWARDS][new_index]

                if nei_obs is not None:
                    obs_list.append(nei_obs)
                    act_list.append(nei_act)
                    region_obs_act_step.append(np.concatenate([nei_obs, nei_act]))
                    region_reward_step.append(nei_reward)

            if len(region_obs_act_step) < self.config["num_region_agents"] + 1:
                num_pad = self.config["num_region_agents"] + 1 - len(region_obs_act_step)
                for _ in range(num_pad):
                    region_obs_act_step.append(np.zeros(len(ego_obs_act), dtype='float32'))

            region_obs_act.append(region_obs_act_step)
            region_reward.append(sum(region_reward_step) / len(region_reward_step))


        sample_batch["region_obs_act"] = np.array(region_obs_act)
        sample_batch["region_rewards"] = np.array(region_reward)
        sample_batch["ego_rewards"] = sample_batch["rewards"]
        sample_batch["rewards"] = sample_batch["region_rewards"] + sample_batch["ego_rewards"]

        return sample_batch

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        with (torch.no_grad()):
            if episode is not None:
                self.region_process(sample_batch, other_agent_batches)
            else:
                dim_obs = self.model.dim_obs
                dim_action = self.model.dim_action
                num_region_agents = self.config["num_region_agents"]
                sample_batch["region_obs_act"] = np.zeros((sample_batch.count, num_region_agents, dim_obs + dim_action))
                sample_batch["region_rewards"] = np.zeros(sample_batch.count)
                sample_batch["ego_rewards"] = np.zeros(sample_batch.count)

            sample_batch[SampleBatch.VF_PREDS] = self.model.all_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)
            sample_batch["ego_values"] = self.model.ego_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)
            sample_batch["target_ego_values"] = self.model.target_ego_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)


            sample_batch["region_values"] = self.model.region_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)
            sample_batch["target_region_values"] = self.model.target_region_value_function(
                convert_to_torch_tensor(sample_batch, self.device)
            ).cpu().detach().numpy().astype(np.float32)


            if sample_batch[SampleBatch.DONES][-1]:
                last_ego_r = 0.0
                last_region_r = 0.0
                last_r = 0.0
            else:
                last_ego_r = sample_batch["ego_values"][-1]
                last_region_r = sample_batch["region_values"][-1]
                last_r = sample_batch[SampleBatch.VF_PREDS][-1]

            sample_batch = compute_advantages(
                sample_batch,
                last_r,
                self.config["gamma"],
                self.config["lambda"],
                use_gae=self.config["use_gae"],
                use_critic=self.config.get("use_critic", True)
            )
            sample_batch = compute_ego_advantage(sample_batch, last_ego_r, self.config["gamma"], self.config["lambda"])
            sample_batch = compute_region_advantage(
                sample_batch, last_region_r, self.config["gamma"], self.config["lambda"]
            )

        return sample_batch

    def optimizer(self, ):
        actor_optimizer = torch.optim.Adam(
            params=self.model.actor_variables(), lr=self.config["actor_lr"], eps=1e-7
        )
        ego_critic_optimizer = torch.optim.Adam(
            params=self.model.ego_critic_variables(), lr=self.config["ego_critic_lr"], eps=1e-7
        )
        region_critic_optimizer = torch.optim.Adam(
            params=self.model.region_critic_variables(), lr=self.config["region_critic_lr"], eps=1e-7
        )
        return [ego_critic_optimizer, region_critic_optimizer, actor_optimizer]

    def apply_gradients(self, gradients: ModelGradients) -> None:
        super(MAAPOPolicy, self).apply_gradients(gradients)
        if self.num_grad_updates % self.target_model_update_freq == 0:
            self.model.ego_soft_update()
            self.model.region_soft_update()
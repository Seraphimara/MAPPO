import torch
import torch.nn as nn



class AttentionCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, embed_dim=128, num_heads=4, norm_in=False):
        super(AttentionCritic, self).__init__()
        assert (embed_dim % num_heads) == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        encoder = nn.Sequential()
        if norm_in:
            encoder.add_module('enc_bn', nn.BatchNorm1d(obs_dim + action_dim, affine=False))
        encoder.add_module('enc_fc1', nn.Linear(obs_dim + action_dim, embed_dim))
        encoder.add_module('enc_nl', nn.LeakyReLU())
        self.encoder = encoder
        
        state_encoder = nn.Sequential()
        if norm_in:
            state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(obs_dim, affine=False))
        state_encoder.add_module('s_enc_fc1', nn.Linear(obs_dim, embed_dim))
        state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
        self.state_encoder = state_encoder

        critic = nn.Sequential()
        critic.add_module('critic_fc1', nn.Linear(2 * embed_dim, embed_dim))
        critic.add_module('critic_nl', nn.LeakyReLU())
        critic.add_module('critic_fc2', nn.Linear(embed_dim, 1))
        self.critic = critic
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, inputs):
        key = self.encoder(inputs)
        ego_obs = inputs[:, :, 0:self.obs_dim]
        query = self.state_encoder(ego_obs)
        value = self.encoder(inputs)
        output, _ = self.multihead_attn(query, key, value)
        critic_in = torch.cat([query[:, 0, :], output[:, 0, :]], dim=1)
        q_value = torch.reshape(self.critic(critic_in), [-1])
        return q_value


if __name__ == "__main__":
    inputs = torch.randn(4, 5, 10)
    test_critic = AttentionCritic(8,2,num_heads=4)
    ret = test_critic(inputs)

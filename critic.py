import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import multi_head_attention

class attention_critic(nn.Module):
    def __init__(self, num_agent, sa_dims, s_dims, head_dim, output_dim, head_num=4):
        super(attention_critic, self).__init__()
        self.num_agent = num_agent
        self.sa_dims = sa_dims
        self.s_dims = s_dims
        self.output_dim = output_dim
        self.head_num = head_num
        # * Note: hidden_dim % num_agent == 0
        self.hidden_dim = head_dim * self.head_num
        self.attention_unit = multi_head_attention(head_num=self.head_num, model_dim=self.hidden_dim)

        self.sa_encoder_layer = [nn.Sequential(nn.BatchNorm1d(self.sa_dims[i], affine=False), nn.Linear(self.sa_dims[i], self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)) for i in range(self.num_agent)]
        self.critic_layer = [nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.output_dim[i])) for i in range(self.num_agent)]
        self.s_encoder_layer = [nn.Sequential(nn.BatchNorm1d(self.s_dims[i], affine=False), nn.Linear(self.s_dims[i], self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.hidden_dim)) for i in range(self.num_agent)]

        gain = nn.init.calculate_gain('leaky_relu')
        for i in range(self.num_agent):
            for layer in self.sa_encoder_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
            for layer in self.critic_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
            for layer in self.s_encoder_layer:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)

    def forward(self, s_inputs, a_inputs, all=False, reg=False):
        sa_embedding_features = []
        s_embedding_features = []
        for i in range(self.num_agent):
            sa_embedding_features.append(F.leaky_relu(self.sa_encoder_layer[i](torch.cat([s_inputs[i], a_inputs[i]], dim=1))).unsqueeze(1))
            s_embedding_features.append(F.leaky_relu(self.s_encoder_layer[i](s_inputs[i])).unsqueeze(1))
        x, reg_atten = self.attention_unit.forward(torch.cat(sa_embedding_features, dim=1), torch.cat(s_embedding_features, dim=1))
        all_q_values = []
        q_values = []
        for i in range(self.num_agent):
            all_q_values.append(self.critic_layer[i](torch.cat([x[:, i, :].squeeze(1), sa_embedding_features[i].squeeze(1)], dim=1)))

        for i in range(self.num_agent):
            max_actions = all_q_values[i].max(dim=1, keepdim=True)[1]
            q_values.append(all_q_values[i].gather(1, max_actions))

        res = [q_values]
        if all:
            res.append(all_q_values)
        if reg:
            res.append(reg_atten)
        if len(res) > 1:
            return tuple(res)
        else:
            return res[0]

    def get_shared_parameters(self):
        parameters = self.attention_unit.parameters()
        return parameters
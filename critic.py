import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import multi_head_attention
from itertools import chain

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
        self.attention_unit = multi_head_attention(head_num=self.head_num, model_dim=self.hidden_dim, num_agent=self.num_agent)

        self.sa_encoder_layer = nn.ModuleList(nn.Sequential(nn.BatchNorm1d(self.sa_dims[i], affine=False), nn.Linear(self.sa_dims[i], self.hidden_dim), nn.LeakyReLU()) for i in range(self.num_agent))
        self.critic_layer = nn.ModuleList(nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.output_dim[i])) for i in range(self.num_agent))
        self.s_encoder_layer = nn.ModuleList(nn.Sequential(nn.BatchNorm1d(self.s_dims[i], affine=False), nn.Linear(self.s_dims[i], self.hidden_dim), nn.LeakyReLU()) for i in range(self.num_agent))

        self.shared_modules = [self.attention_unit.Q_projs, self.attention_unit.K_projs, self.attention_unit.V_projs, self.sa_encoder_layer]


    def forward(self, s_inputs, a_inputs, all=False, reg=False):
        sa_embedding_features = []
        s_embedding_features = []
        for i in range(self.num_agent):
            sa_embedding_features.append(self.sa_encoder_layer[i](torch.cat([s_inputs[i], a_inputs[i]], dim=1)))
            s_embedding_features.append(self.s_encoder_layer[i](s_inputs[i]))
        all_atten_values, reg_atten = self.attention_unit.forward(sa_embedding_features, s_embedding_features)
        all_q_values = []
        q_values = []
        for i in range(self.num_agent):
            all_q_values.append(self.critic_layer[i](torch.cat([s_embedding_features[i], *all_atten_values[i]], dim=1)))

        for i in range(self.num_agent):
            actions = a_inputs[i].max(dim=1, keepdim=True)[1]
            q_values.append(all_q_values[i].gather(1, actions))

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
        return chain(*[m.parameters() for m in self.shared_modules])
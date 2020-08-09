import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class multi_head_attention(nn.Module):
    def __init__(self, head_num, model_dim, num_agent):
        super(multi_head_attention, self).__init__()
        self.head_num = head_num
        self.model_dim = model_dim
        self.head_dim = self.model_dim // self.head_num
        self.num_agent = num_agent

        self.Q_projs = nn.ModuleList(nn.Sequential(
            nn.Linear(self.model_dim, self.head_dim, bias=False),
        ) for _ in range(self.head_num))
        self.K_projs = nn.ModuleList(nn.Sequential(
            nn.Linear(self.model_dim, self.head_dim, bias=False),
        ) for _ in range(self.head_num))
        self.V_projs = nn.ModuleList(nn.Sequential(
            nn.Linear(self.model_dim, self.head_dim),
            nn.LeakyReLU()
        ) for _ in range(self.head_num))

    def forward(self, sa_inputs, s_inputs):
        all_qs = [[head(s_input) for s_input in s_inputs] for head in self.Q_projs]
        all_ks = [[head(sa_input) for sa_input in sa_inputs] for head in self.K_projs]
        all_vs = [[head(sa_input) for sa_input in sa_inputs] for head in self.V_projs]
        scale = self.head_dim ** -0.5

        all_atten_values = [[] for _ in range(self.num_agent)]
        reg_atten = 0
        for cur_head_qs, cur_head_vs, cur_head_ks in zip(all_qs, all_vs, all_ks):
            for agent_idx, q in zip(range(self.num_agent), cur_head_qs):
                ks = [k for idx, k in enumerate(cur_head_ks) if idx != agent_idx]
                vs = [v for idx, v in enumerate(cur_head_vs) if idx != agent_idx]
                logits = torch.matmul(q.view(q.size(0), 1, -1), torch.stack(ks).permute(1, 2, 0))
                scale_logits = scale * logits
                scale_dot_product = F.softmax(scale_logits, dim=2)
                other_values = (scale_dot_product * torch.stack(vs).permute(1, 2, 0)).sum(dim=2)
                all_atten_values[agent_idx].append(other_values)
                reg_atten += logits.mean() * 1e-3
        return all_atten_values, reg_atten
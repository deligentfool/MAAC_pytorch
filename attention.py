import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(scaled_dot_product_attention, self).__init__()

    def forward(self, Q, K, V, scale=None, self_mask=True):
        attention = torch.bmm(Q, K.permute(0, 2, 1))
        reg_atten = (attention ** 2).mean()
        if scale:
            attention = attention * scale
        if self_mask:
            mask = torch.eye(attention.size(1), attention.size(2))
            mask = mask.unsqueeze(0).expand([attention.size(0), mask.size(0), mask.size(1)])
            attention.masked_fill_(mask=mask.bool(), value=-1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.bmm(attention, V)
        return context, reg_atten


class multi_head_attention(nn.Module):
    def __init__(self, head_num, model_dim):
        super(multi_head_attention, self).__init__()
        self.head_num = head_num
        self.model_dim = model_dim
        self.head_dim = self.model_dim // self.head_num

        self.Q_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim, bias=False),
        )
        self.K_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim, bias=False),
        )
        self.V_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LeakyReLU()
        )
        self.attention = scaled_dot_product_attention()

    def forward(self, sa_input, s_input):
        Q = self.Q_proj(s_input)
        K = self.K_proj(sa_input)
        V = self.V_proj(sa_input)
        Q = Q.view(Q.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        Q = Q.contiguous().view(Q.size(0) * self.head_num, -1, self.head_dim)
        K = K.view(K.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        K = K.contiguous().view(K.size(0) * self.head_num, -1, self.head_dim)
        V = V.view(V.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        V = V.contiguous().view(V.size(0) * self.head_num, -1, self.head_dim)
        scale = K.size(-1) ** -0.5

        context, reg_atten = self.attention.forward(Q, K, V, scale=scale)
        context = context.view(sa_input.size(0), self.head_num, -1, self.head_dim).transpose(1, 2)
        context = context.contiguous().view(sa_input.size(0), -1, self.model_dim)
        return context, reg_atten
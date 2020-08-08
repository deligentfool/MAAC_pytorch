import torch
import torch.nn as nn
import torch.nn.functional as F


class discrete_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(discrete_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.fc1_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_layer = nn.Linear(self.hidden_dim, self.output_dim)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.fc1_layer.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc2_layer.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc3_layer.weight, gain=gain)

    def forward(self, input, explore=True, mask=None, log=False, reg=False, entropy=False, all=False):
        x = F.leaky_relu(self.fc1_layer(input))
        x = F.leaky_relu(self.fc2_layer(x))
        x = self.fc3_layer(x)
        res = []
        prob = F.softmax(x, dim=-1)
        if mask is not None:
            prob.masked_fill_(mask, value=0.0)
        if explore:
            action = torch.multinomial(prob, 1)
        else:
            action = prob.max(1, keepdim=True)[1]
        res = [action]
        if reg:
            reg_policy = torch.mean(x ** 2)
            res.append(reg_policy)
        if log:
            res.append(F.log_softmax(x, -1).gather(1, action))
        if entropy:
            res.append(-(F.log_softmax(x, dim=-1) * F.softmax(x, dim=-1)).sum(1).mean())
        if all:
            res.append(F.softmax(x, dim=-1))
        if len(res) > 1:
            return tuple(res)
        else:
            return res[0]
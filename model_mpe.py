import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import discrete_policy_net
from critic import attention_critic
import numpy as np
from buffer import replay_buffer
from utils import make_env
import argparse
import os
import random
from gym.spaces.discrete import Discrete


class maac(object):
    def __init__(self, env_id, batch_size, learning_rate, exploration, episode, gamma, alpha, capacity, rho, update_iter, update_every, head_dim, traj_len, render):
        self.env_id = env_id
        self.env = make_env(self.env_id, discrete_action=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.episode = episode
        self.gamma = gamma
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.update_every = update_every
        self.head_dim = head_dim
        self.traj_len = traj_len
        self.render = render

        self.observation_dims = [self.env.observation_space[i].shape[0] for i in range(self.env.n)]
        self.action_dims = [self.env.action_space[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape for i in range(self.env.n)]

        self.alphas = [alpha for _ in range(self.env.n)]

        self.value_net = attention_critic(num_agent=self.env.n, sa_dims=[o + a for o, a in zip(self.observation_dims, self.action_dims)], s_dims=self.observation_dims, head_dim=self.head_dim, output_dim=self.action_dims)
        self.target_value_net = attention_critic(num_agent=self.env.n, sa_dims=[o + a for o, a in zip(self.observation_dims, self.action_dims)], s_dims=self.observation_dims, head_dim=self.head_dim, output_dim=self.action_dims)
        self.policy_nets = [discrete_policy_net(input_dim=self.observation_dims[n], output_dim=self.action_dims[n]) for n in range(self.env.n)]
        self.target_policy_nets = [discrete_policy_net(input_dim=self.observation_dims[n], output_dim=self.action_dims[n]) for n in range(self.env.n)]
        [self.target_policy_nets[n].load_state_dict(self.policy_nets[n].state_dict()) for n in range(self.env.n)]
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizers = [torch.optim.Adam(self.policy_nets[n].parameters(), lr=self.learning_rate) for n in range(self.env.n)]

        self.count = 0
        self.train_count = 0

    def soft_value_update(self):
        for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def soft_policy_update(self, policy_idx):
        for param, target_param in zip(self.policy_nets[policy_idx].parameters(), self.target_policy_nets[policy_idx].parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def train(self):
        for _ in range(self.update_iter):
            observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

            indiv_observations = [torch.FloatTensor(np.vstack([observations[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]
            indiv_actions = [torch.FloatTensor([actions[b][n] for b in range(self.batch_size)]) for n in range(self.env.n)]
            one_hot_indiv_actions = [torch.zeros(self.batch_size, self.action_dims[n]) for n in range(self.env.n)]
            one_hot_indiv_actions =[one_hot_indiv_actions[n].scatter(dim=1, index=indiv_actions[n].unsqueeze(1).long(), value=1) for n in range(self.env.n)]
            rewards = torch.FloatTensor(rewards)
            indiv_rewards = [rewards[:, n] for n in range(self.env.n)]
            indiv_next_observations = [torch.FloatTensor(np.vstack([next_observations[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]
            dones = torch.FloatTensor(dones)
            indiv_dones = [dones[:, n] for n in range(self.env.n)]

            one_hot_next_actions = []
            next_actions = []
            next_log_policies = []
            for i in range(self.env.n):
                next_action, next_log_policy = self.target_policy_nets[i].forward(indiv_next_observations[i], log=True)
                next_log_policies.append(next_log_policy)
                next_actions.append(next_action)
                one_hot_next_action = torch.zeros(self.batch_size, self.action_dims[i])
                one_hot_next_action.scatter_(dim=1, index=next_action, value=1)
                one_hot_next_actions.append(one_hot_next_action)
            next_q = self.target_value_net.forward(indiv_next_observations, one_hot_next_actions)
            q, reg_atten = self.value_net.forward(indiv_observations, one_hot_indiv_actions, reg=True)
            value_loss = 0
            for i in range(self.env.n):
                # * calculate the expectation directly
                target_q = indiv_rewards[i].unsqueeze(1) + (1 - indiv_dones[i].unsqueeze(1)) * self.gamma * (next_q[i] - self.alphas[i] * next_log_policies[i])
                target_q = target_q.detach()

                value_loss += (q[i] - target_q).pow(2).mean()

            value_loss += 1e-3 * reg_atten
            self.value_optimizer.zero_grad()
            value_loss.backward()
            for p in self.value_net.get_shared_parameters():
                p.grad.data.mul_(1. / self.env.n)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.soft_value_update()

            observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

            indiv_observations = [torch.FloatTensor(np.vstack([observations[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]

            one_hot_sample_actions = []
            sample_actions = []
            log_policies = []
            entropies = []
            all_policies = []
            reg_policies = []
            for i in range(self.env.n):
                #dist = torch.distributions.Categorical(cur_policy)
                sample_action, reg_policy, log_policy, entropy, all_policy  = self.policy_nets[i].forward(indiv_observations[i], explore=True, log=True, reg=True, entropy=True, all=True)
                sample_actions.append(sample_action)
                reg_policies.append(reg_policy)
                log_policies.append(log_policy)
                entropies.append(entropy)
                all_policies.append(all_policy)
                one_hot_sample_action = torch.zeros(self.batch_size, self.action_dims[i])
                one_hot_sample_action.scatter_(dim=1, index=sample_action, value=1)
                one_hot_sample_actions.append(one_hot_sample_action)

            q, all_q = self.value_net(indiv_observations, one_hot_sample_actions, all=True)
            for i in range(self.env.n):
                b = torch.sum(all_policies[i] * all_q[i], dim=1, keepdim=True).detach()
                adv = (q[i] - b).detach()
                policy_loss = log_policies[i] * (self.alphas[i] * log_policies[i] - adv).detach()
                policy_loss = policy_loss.mean() + reg_policies[i] * 1e-3

                self.policy_optimizers[i].zero_grad()
                for p in self.value_net.parameters():
                    p.requires_grad = False
                policy_loss.backward()
                for p in self.value_net.parameters():
                    p.requires_grad = True
                nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.policy_optimizers[i].step()

                self.soft_policy_update(i)

    def run(self):
        max_reward = -np.inf
        weight_reward = [None for i in range(self.env.n)]
        for epi in range(self.episode):
            self.env.reset()
            if self.render:
                self.env.render()
            total_reward = [0 for i in range(self.env.n)]
            obs = self.env.reset()
            while True:
                action_indice = []
                actions = []
                for i in range(self.env.n):
                    if epi >= self.exploration:
                        action_idx = self.policy_nets[i].forward(torch.FloatTensor(np.expand_dims(obs[i], 0)), explore=True).item()
                    else:
                        action_idx = np.random.choice(list(range(self.action_dims[i])))
                    action = np.zeros(self.action_dims[i])
                    action[action_idx] = 1
                    actions.append(action)
                    action_indice.append(action_idx)
                next_obs, reward, done, _ = self.env.step(actions)
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action_indice, reward, next_obs, done)
                self.count += 1
                total_reward = [tr + r for tr, r in zip(total_reward, reward)]
                obs = next_obs

                if (self.count % self.update_every) == 0 and epi >= self.exploration:
                    self.train_count += 1
                    self.train()
                if self.count % self.traj_len == 0:
                    done = [True for _ in range(self.env.n)]
                if any(done):
                    if weight_reward[0] is None:
                        weight_reward = total_reward
                    else:
                        weight_reward = [wr * 0.99 + tr * 0.01 for wr, tr in zip(weight_reward, total_reward)]
                    if sum(weight_reward) > max_reward and epi >= self.exploration:
                        torch.save(self.value_net, './models/{}/value.pkl'.format(self.env_id))
                        for i in range(self.env.n):
                            torch.save(self.policy_nets[i], './models/{}/policy{}.pkl'.format(self.env_id, i))
                        max_reward = sum(weight_reward)
                    print(('episode: {}\ttrain_count:{}\tweight_reward:' + '{:.2f}\t' * self.env.n).format(epi + 1, self.train_count, *weight_reward))
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scenario",
        help="set the scenario.",
        type=str,
        default='fullobs_collect_treasure'
    )
    args = parser.parse_args()
    env_id = args.scenario
    os.makedirs('./models/{}'.format(env_id), exist_ok=True)
    # * the size of replay buffer must be appropriate
    test = maac(
        env_id=env_id,
        batch_size=256,
        learning_rate=1e-4,
        exploration=200,
        episode=100000,
        gamma=0.96,
        alpha=0.01,
        capacity=256,
        rho=0.995,
        update_iter=1,
        update_every=50,
        head_dim=32,
        traj_len=45,
        render=False
    )
    test.run()
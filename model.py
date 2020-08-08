import torch
import torch.nn as nn
import torch.nn.functional as F
from policy import discrete_policy_net
from critic import attention_critic
import numpy as np
from buffer import replay_buffer
from smac.env import StarCraft2Env
import argparse
import os
import random


class maac(object):
    def __init__(self, env_id, batch_size, learning_rate, exploration, episode, gamma, alpha, auto_entropy_tuning, capacity, rho, update_iter, update_every, head_dim):
        self.env_id = env_id
        self.env = StarCraft2Env(map_name=env_id)
        self.env_info = self.env.get_env_info()
        self.num_agent = self.env_info['n_agents']
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.episode = episode
        self.gamma = gamma
        self.auto_entropy_tuning = auto_entropy_tuning
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.update_every = update_every
        self.head_dim = head_dim

        self.observation_dims = [self.env_info['obs_shape'] for i in range(self.num_agent)]
        self.action_dims = [self.env_info['n_actions'] for i in range(self.num_agent)]

        if not self.auto_entropy_tuning:
            self.alphas = [alpha for _ in range(self.num_agent)]
        else:
            self.log_alphas = [torch.zeros(1, requires_grad=True) for _ in range(self.num_agent)]
            self.alphas = [self.log_alphas[n].exp() for n in range(self.num_agent)]
            # * set the max possible entropy as the target entropy
            self.target_entropys = [-np.log((1. / self.action_dims[n])) * 0.98 for n in range(self.num_agent)]
            self.alpha_optimizers = [torch.optim.Adam([self.log_alphas[n]], lr=self.learning_rate, eps=1e-4) for n in range(self.num_agent)]

        self.value_net = attention_critic(num_agent=self.num_agent, sa_dims=[o + a for o, a in zip(self.observation_dims, self.action_dims)], s_dims=self.observation_dims, head_dim=self.head_dim, output_dim=self.action_dims)
        self.target_value_net = attention_critic(num_agent=self.num_agent, sa_dims=[o + a for o, a in zip(self.observation_dims, self.action_dims)], s_dims=self.observation_dims, head_dim=self.head_dim, output_dim=self.action_dims)
        self.policy_nets = [discrete_policy_net(input_dim=self.observation_dims[n], output_dim=self.action_dims[n]) for n in range(self.num_agent)]
        self.target_policy_nets = [discrete_policy_net(input_dim=self.observation_dims[n], output_dim=self.action_dims[n]) for n in range(self.num_agent)]
        [self.target_policy_nets[n].load_state_dict(self.policy_nets[n].state_dict()) for n in range(self.num_agent)]
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.policy_optimizers = [torch.optim.Adam(self.policy_nets[n].parameters(), lr=self.learning_rate) for n in range(self.num_agent)]

        self.weight_reward = None
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

            indiv_observations = [torch.FloatTensor(np.vstack([observations[b][n] for b in range(self.batch_size)])) for n in range(self.num_agent)]
            indiv_actions = [torch.FloatTensor([actions[b][n] for b in range(self.batch_size)]) for n in range(self.num_agent)]
            one_hot_indiv_actions = [torch.zeros(self.batch_size, self.action_dims[n]) for n in range(self.num_agent)]
            one_hot_indiv_actions =[one_hot_indiv_actions[n].scatter(dim=1, index=indiv_actions[n].unsqueeze(1).long(), value=1) for n in range(self.num_agent)]
            rewards = torch.FloatTensor(rewards)
            indiv_next_observations = [torch.FloatTensor(np.vstack([next_observations[b][n] for b in range(self.batch_size)])) for n in range(self.num_agent)]
            dones = torch.FloatTensor(dones)

            one_hot_next_sample_actions = []
            next_sample_actions = []
            for i in range(self.num_agent):
                policy, _ = self.target_policy_nets[i].forward(indiv_next_observations[i])
                next_sample_action = self.target_policy_nets[i].act(indiv_next_observations[i], explore=True)
                next_sample_actions.append(next_sample_action)
                one_hot_next_sample_action = torch.zeros(self.batch_size, self.action_dims[i])
                one_hot_next_sample_action.scatter_(dim=1, index=next_sample_action, value=1)
                one_hot_next_sample_actions.append(one_hot_next_sample_action)
            total_target_q_value, _ = self.target_value_net.forward(indiv_next_observations, one_hot_next_sample_actions)
            total_q, reg_atten = self.value_net.forward(indiv_observations, one_hot_indiv_actions)
            value_loss = 0
            for i in range(self.num_agent):
                target_q_value = total_target_q_value[i].gather(1, next_sample_actions[i])
                # * calculate the expectation directly
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * (target_q_value - self.alphas[i] * policy.log().gather(1, next_sample_actions[i]))
                target_q = target_q.detach()

                q = total_q[i].gather(dim=1, index=indiv_actions[i].unsqueeze(1).long())
                value_loss += (q - target_q).pow(2).mean()

            value_loss += 1e-3 * reg_atten
            self.value_optimizer.zero_grad()
            value_loss.backward()
            for p in self.value_net.get_shared_parameters():
                p.grad.data.mul_(1. / self.num_agent)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            self.soft_value_update()

            observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

            indiv_observations = [torch.FloatTensor(np.vstack([observations[b][n] for b in range(self.batch_size)])) for n in range(self.num_agent)]
            indiv_actions = [torch.FloatTensor([actions[b][n] for b in range(self.batch_size)]) for n in range(self.num_agent)]
            one_hot_indiv_actions = [torch.zeros(self.batch_size, self.action_dims[n]) for n in range(self.num_agent)]
            one_hot_indiv_actions =[one_hot_indiv_actions[n].scatter(dim=1, index=indiv_actions[n].unsqueeze(1).long(), value=1) for n in range(self.num_agent)]
            rewards = torch.FloatTensor(rewards)
            indiv_next_observations = [torch.FloatTensor(np.vstack([next_observations[b][n] for b in range(self.batch_size)])) for n in range(self.num_agent)]
            dones = torch.FloatTensor(dones)

            one_hot_sample_actions = []
            sample_actions = []
            for i in range(self.num_agent):
                #dist = torch.distributions.Categorical(cur_policy)
                sample_action = self.policy_nets[i].act(indiv_observations[i], explore=True)
                sample_actions.append(sample_action)
                one_hot_sample_action = torch.zeros(self.batch_size, self.action_dims[i])
                one_hot_sample_action.scatter_(dim=1, index=sample_action, value=1)
                one_hot_sample_actions.append(one_hot_sample_action)

            target_q_value, _ = self.value_net(indiv_observations, one_hot_sample_actions)
            for i in range(self.num_agent):
                cur_policy, reg_policy = self.policy_nets[i].forward(indiv_observations[i])
                b = torch.sum(cur_policy * target_q_value[i], dim=1, keepdim=True).detach()
                adv = (target_q_value[i].gather(1, sample_actions[i]) - b).detach()
                policy_loss = cur_policy.log().gather(1, sample_actions[i]) * (self.alphas[i] * cur_policy.log().gather(1, sample_actions[i]) - adv).detach()
                policy_loss = policy_loss.mean() + reg_policy * 1e-3

                self.policy_optimizers[i].zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.policy_optimizers[i].step()

                if self.auto_entropy_tuning:
                    self.alpha_optimizers[i].zero_grad()
                    entropy_loss = -(self.log_alphas[i] * (cur_policy.log() + self.target_entropys[i]).detach()).mean()
                    entropy_loss.backward()
                    nn.utils.clip_grad_norm_([self.log_alphas[i]], 0.2)
                    self.alpha_optimizers[i].step()

                    self.alphas[i] = self.log_alphas[i].exp()

                self.soft_policy_update(i)

    def run(self):
        max_reward = 0
        for epi in range(self.episode):
            self.env.reset()
            total_reward = 0
            obs = self.env.get_obs()
            while True:
                #state = env.get_state()
                actions = []
                for i in range(self.num_agent):
                    avail_actions = self.env.get_avail_agent_actions(i)
                    actions_mask = [0 if tag == 1 else 1 for tag in avail_actions]
                    if epi >= self.exploration:
                        action_idx = self.policy_nets[i].act(torch.FloatTensor(np.expand_dims(obs[i], 0)), explore=True, mask=torch.BoolTensor(actions_mask).unsqueeze(0))
                    else:
                        avail_actions_ind = np.nonzero(avail_actions)[0]
                        action_idx = np.random.choice(avail_actions_ind)
                    actions.append(action_idx)
                reward, done, _ = self.env.step(actions)
                next_obs = self.env.get_obs()
                self.buffer.store(obs, actions, reward, next_obs, done)
                self.count += 1
                total_reward = total_reward + reward
                obs = next_obs

                if (self.count % self.update_every) == 0 and epi >= self.exploration:
                    self.train_count += 1
                    self.train()
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    if self.weight_reward > max_reward and epi >= self.exploration:
                        torch.save(self.value_net, './models/{}/value.pkl'.format(self.env_id))
                        for i in range(self.num_agent):
                            torch.save(self.policy_nets[i], './models/{}/policy{}.pkl'.format(self.env_id, i))
                        max_reward = self.weight_reward
                    print('episode: {}\ttrain_count:{}\tweight_reward:{:.2f}'.format(epi + 1, self.train_count, self.weight_reward))
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scenario",
        help="set the scenario.",
        type=str,
        default='3m'
    )
    args = parser.parse_args()
    env_id = args.scenario
    os.makedirs('./models/{}'.format(env_id), exist_ok=True)
    # * the size of replay buffer must be appropriate
    test = maac(
        env_id=env_id,
        batch_size=256,
        learning_rate=1e-3,
        exploration=20,
        episode=10000,
        gamma=0.99,
        alpha=0.01,
        auto_entropy_tuning=False,
        capacity=50000,
        rho=0.995,
        update_iter=1,
        update_every=50,
        head_dim=32
    )
    test.run()
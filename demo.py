"""
email: zaifyahsan@gmail.com
(c) Faizy Ahsan
"""

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        # print('state:', state.shape)
        pi1 = F.relu(self.pi1(state))
        # print('pi1:', pi1.shape)
        v1 = F.relu(self.v1(state))
        # print('v1:', v1.shape)

        pi = self.pi(pi1)
        # print('pi:', pi.shape)
        v = self.v(v1)
        # print('v:', v.shape)

        return pi, v

    def calc_R(self, done):
        var_states = []
        for i, item in enumerate(self.states):
            while isinstance(item, tuple):
                item = item[0]
            var_states.append(item)
        self.states = var_states

        states = T.tensor(np.array(self.states), dtype=T.float)

        # states = T.tensor(np.array([item[0] for item in self.states]), dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        var_states = []
        for i, item in enumerate(self.states):
            while isinstance(item, tuple):
                item = item[0]
            var_states.append(item)
        self.states = var_states

        states = T.tensor(np.array(self.states), dtype=T.float)
        # print('calc_loss self.actions:', self.actions); exit(0)
        actions = T.tensor(np.array(self.actions), dtype=T.float)
        # print('pass till here')
        # print('calc_loss done:', done)
        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        # print('before observation:', observation)
        # print('shape observation:', observation.shape)
        # observation = observation[0]
        # print('observation:', observation)
        # print('shape observation:', observation.shape); exit(0)

        while isinstance(observation, tuple):
            observation = observation[0]

        state = T.tensor(np.array([observation]), dtype=T.float)
        # print('state:', state)
        # print('shape state:', state.shape)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        # print('action:', action)

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_id, N_GAMES, T_MAX):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.N_GAMES = N_GAMES
        self.T_MAX = T_MAX

    def run(self):
        t_step = 1
        while self.episode_idx.value < self.N_GAMES:
            done = False
            observation = self.env.reset()
            observation = (observation, {})

            # print('observation:', observation); exit(0)
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                # print('observation:', observation); exit(0)
                action = self.local_actor_critic.choose_action(observation)
                # print('action:', action); exit(0)
                # observation_, reward, done, info, _ = self.env.step(action)
                # print('check:', self.env.step(action)); exit(0)
                # observation_, reward, done, info = self.env.step(action)

                collect_from_action = self.env.step(action)
                observation_ = collect_from_action[0]
                reward = collect_from_action[1]
                done = collect_from_action[2]
                info = collect_from_action[3]

                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % self.T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    # print('loss:', loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # print('after loss:', loss)
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                # print('observation_:', observation_); exit(0)
                observation = (observation_, {})
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)


if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v1'
    n_actions = 2
    input_dims = [4]
    N_GAMES = 3000
    T_MAX = 5
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                       betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id,
                     N_GAMES=N_GAMES,
                     T_MAX = T_MAX) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]



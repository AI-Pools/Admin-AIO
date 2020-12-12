import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple
from sklearn.utils import shuffle
from torch.autograd import Variable

STATE_SPACE = 2
ACTION_SPACE = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=STATE_SPACE, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=20)
        self.out = nn.Linear(in_features=20, out_features=ACTION_SPACE)

    def forward(self, t):
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t

class Memory():
    def __init__(self, size):
        self.size = size
        self.sequence = namedtuple("sequence", ["state", "action", "reward", "new_state"])
        self.memory = []

    def add(self, state, action, reward, new_state):
        if len(self.memory) > self.size:
            self.memory = self.memory[1:]
        self.memory.append(self.sequence(state, action, reward, new_state))

    def sample(self, sample_size):
        index = np.random.randint(0, self.size - sample_size - 1)
        sample = self.memory[index:index+sample_size]

        sample_state = [s.state for s in sample]
        sample_action = [s.action for s in sample]
        sample_reward = [s.reward for s in sample]
        sample_new_state = [s.new_state for s in sample]

        return sample_state, sample_action, sample_reward, sample_new_state

class Agent():
    def __init__(self):
        self.model = Network()
        self.target_model = Network()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.target_update_counter = 0
        self.memory = Memory(50_000)

    def get_qs(self, state):
        return self.model(state)

    def train(self, terminal_state, step):
        states, actions, rewards, next_states = self.memory.sample(32)
        curr_qs = self.model(torch.tensor(states).float())
        next_qs = self.model(torch.tensor(next_states).float())

        X = []
        y = []

        for index in range(len(states)):
            new_q = rewards[index] + torch.max(next_qs[index]) * GAMMA
            current_qs = curr_qs[index]
            current_qs[actions[index]] = new_q

            X.append(states[index])
            y.append(current_qs)

        predictions = self.model(torch.tensor(X).float())

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > 100:
            self.target_model.load_state_dict(model.state_dict())
            self.target_update_counter = 0

agent = Agent()
env = gym.make("MountainCar-v0")
state = env.reset()
epsilon = 1

for episode in range(1_000):
    done = False
    state = env.reset()
    episode_reward = []
    while not done:
        if episode % 1_000 == 0 and episode > 0:
            env.render()
        if np.random.rand() > epsilon:
            action = torch.argmax(agent.get_qs(torch.tensor(state).float())).item()
        else:
            action = np.random.randint(0, ACTION_SPACE)
        new_state, reward, done, _ = env.step(action)
        episode_reward.append(reward)
        agent.memory.add(state, action, reward, new_state)
        state = new_state
        if len(agent.memory.memory) == agent.memory.size:
            agent.train(done, 0)

    epsilon = max(0.1, epsilon * 0.99)

    if episode % 50 == 0:
        print(f"episode {episode} mean:{np.mean(episode_reward)}")
        



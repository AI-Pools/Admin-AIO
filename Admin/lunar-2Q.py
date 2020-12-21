import gym
import gym_chrome_dino
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.utils import shuffle
from collections import namedtuple

STATE_SPACE = 8
ACTION_SPACE = 4

BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99

EPS_MIN = 0.1
EPS_DECAY = 0.99

"""Agent network"""
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SPACE, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.out = nn.Linear(50, ACTION_SPACE)

    def forward(self, t):
        if len(t.shape) == 3:
            t = t.unsqueeze(0)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.fc3(t)
        t = F.relu(t)
        return self.out(t)

"""Agent memory"""
class Memory():
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.sequence = namedtuple("sequence", ["state", "action", "reward", "new_state"])

    def add(self, state, action, reward, new_state):
        if len(self.memory) == self.size:
            self.memory = self.memory[1:]
        self.memory.append(self.sequence(state, action, reward, new_state))

    def shuffle(self):
        self.memory = shuffle(self.memory)

"""RL agent"""
class Agent():
    def __init__(self, network=Network, memory_size=25_000, update_target_every=1_500):
        self.network = Network()
        self.target_network = Network()
        self.target_network.load_state_dict(self.network.state_dict())
        self._update_target_counter = 0
        self.update_target_every = update_target_every
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.mse_loss = nn.MSELoss()
        self.memory = Memory(memory_size)

    """Use epsilon to pick action"""
    def pick_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, ACTION_SPACE)
        else:
            action = torch.argmax(self.network(state)).item()
        return action

    """Calc expectation"""
    def __calc_target(self, reward, next_state):
        if next_state is None:
            return reward
        preds = self.target_network(torch.tensor(next_state).float())
        best_q = preds.max(dim=0)[0].item()
        return reward + GAMMA * best_q

    """train agent over one batch"""
    def __train_in_batch(self, states, actions, rewards, next_states):
        targets = [self.__calc_target(r, nxt) for r, nxt in zip(rewards, next_states)]
        if states.shape[0] != BATCH_SIZE:
            return
        self.optimizer.zero_grad()

        predictions = self.network(states)

        expectations = predictions.data.numpy().copy()
        expectations[range(BATCH_SIZE), actions.int()] = targets
        expectations = torch.tensor(expectations)

        loss_v = self.mse_loss(predictions, expectations)
        loss_v.backward()
        self.optimizer.step()

    """train the agent over the whole memory"""
    def train(self):
        states = torch.tensor([seq.state for seq in self.memory.memory]).float()
        actions =torch.tensor([seq.action for seq in self.memory.memory]).float()
        rewards = torch.tensor([seq.reward for seq in self.memory.memory]).float()
        next_states = [seq.new_state for seq in self.memory.memory]

        for index in range(0, len(self.memory.memory), BATCH_SIZE):
            self._update_target_counter += 1 * BATCH_SIZE
            self.__train_in_batch(states[index: index + BATCH_SIZE], actions[index: index + BATCH_SIZE], rewards[index: index + BATCH_SIZE], next_states[index: index + BATCH_SIZE])

        if self._update_target_counter >= self.update_target_every:
            self.target_network.load_state_dict(self.network.state_dict())
            

env = gym.make("LunarLander-v2")
env.seed(1)
state = env.reset()

def train(agent):
    epsilon = 0.1
    memory = []
    episode_rewards = []
    for episode in range(2_101):
        step = 0
        new_state = env.reset()
        done = False

        for i in range(500):
            # if episode % 50 == 0:
            env.render()
            state = new_state
            action = agent.pick_action(torch.tensor(new_state).unsqueeze(0).float(), epsilon)
            new_state, reward, done, _ = env.step(action)
        
            if done == True:
                print(f"episode: {episode} reward: {sum(episode_rewards):.2f}, epsilon {epsilon:.2f}, iteration {i}")
                episode_rewards = []
                if i != 199:
                    reward = -5
                if i == 199:
                    reward= 2
                if i == 399:
                    reward= 5
                agent.memory.add(state, action, reward, None)
                episode_rewards.append(reward)
                break
            agent.memory.add(state, action, reward, new_state)
            episode_rewards.append(reward)
            if len(memory) > 25_000:
                memory = memory[1:]
            step += 1

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)       

        # if episode % 10 == 0 and episode > 0:
            # agent.memory.shuffle()
            # agent.train()

        # if episode % 100 == 0:
            # torch.save(agent.network.state_dict(), f"lunar_models/lunar-2Q_model_{episode}.torch")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = Agent()

#agent.network = nn.DataParallel(agent.network)
agent.network.load_state_dict(torch.load("./lunar_models/lunar-2Q_model_1200.torch", map_location=torch.device('cpu')))
agent.network.to(device)

train(agent)

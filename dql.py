import gym
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

env = gym.make("MountainCar-v0")
state = env.reset()

model = Network()
sequence = namedtuple("sequence", ["state", "action", "reward", "new_state"])
memory = []
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def model_loss(states, actions, Qtargets):
    # print("state:", states.shape)
    # print("Qtarg:", Qtargets.shape, Qtargets)
    # print("a123")
    preds = model(states.float())
    preds, _ = torch.max(preds, dim=1)
    # preds = torch.tensor([p[torch.argmax(p)] for p in preds]).float()
    # print("b456")
    # print("preds:", preds.shape)
    subs = torch.subtract(preds, Qtargets)
    # print("c")
    sqr = torch.square(subs)
    # print("d")
    # print(sqr[0])
    # print(actions.shape)
    mul = sqr * actions # -> Pourquoi action est de shape [32] et non pas [32][3] ?
    # print("M", mul[0])
    # print("e")
    mn = torch.mean(mul)
    # print("f")
    return Variable(mn, requires_grad = True)
    # return mn
    # return torch.mean(torch.square(torch.subtract(preds, Qtargets)) * actions)
    # tmp = torch.mean(torch.square(torch.subtract(model(states.float()), Qtargets)) * actions)
    # print("g")
    # return torch.mean(torch.square(torch.subtract(model(states.float()), Qtargets)) * actions)

def pick_action(state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(ACTION_SPACE - 1)
    else:
        action = torch.argmax(model(state)).item()

    return action


def train_model(memory):
    states = torch.tensor([seq.state for seq in memory])
    actions =torch.tensor([seq.action for seq in memory]) # TODO: stocker le mask plutot
    rewards = torch.tensor([seq.reward for seq in memory])
    next_states = torch.tensor([seq.new_state for seq in memory])
    losses = []

    # Q_stp1 = model(next_states.float())
    # print("stp1 (out)", Q_stp1.shape, Q_stp1[:10])
    # Qtargets = torch.add(torch.argmax(Q_stp1, dim=1) * GAMMA, rewards) # correspond au reward associé à cette action
    preds = model(next_states.float())
    input_max, _ = torch.max(preds, dim=1)
    Qtargets = torch.add(input_max * GAMMA, rewards) # correspond au reward associé à cette action
    # print("Qt (out)", Qtargets.shape)

    for index in range(0, len(memory), BATCH_SIZE):
        optimizer.zero_grad()
        loss = model_loss(states[index: index + BATCH_SIZE], actions[index: index + BATCH_SIZE], Qtargets[index: index + BATCH_SIZE])
        loss.backward()
        optimizer.step()

def train():
    epsilon = 1
    memory = []
    episode_rewards = []
    for episode in range(50_000):
        step = 0
        state = env.reset()
        done = False

        while done == False:
            if episode % 1_000 == 0 and episode > 0:
                env.render()
            action = pick_action(torch.tensor(state).unsqueeze(0).float(), epsilon)
            new_state, reward, done, _ = env.step(action)
            if reward != -1:
                print("reward:", reward)

            mask = [0] * ACTION_SPACE
            mask[action] = 1

            memory.append(sequence(state, action, reward, new_state))
            episode_rewards.append(reward)
            if len(memory) > 25_000:
                memory = memory[1:]

            state = new_state
            step += 1

            epsilon = max(0.1, epsilon * 0.99)

        if episode % 100 == 0:
            print(f"episode: {episode} mean reward: {np.mean(episode_rewards)}")
            episode_rewards = []

        if episode % 3 == 0 and episode > 0:
            memory = shuffle(memory)
            train_model(memory)
                
train()

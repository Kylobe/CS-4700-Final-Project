import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from collections import deque
import gymnasium as gym
from ChessEnv import ChessEnv
import threading
import sys
import copy


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class AlphaZeroNet(nn.Module):
    def __init__(
            self, env, discount_factor=0.99,
            epsilon_greedy=1.0, epsilon_min=0.05,
            epsilon_decay=0.99, learning_rate=1e-4,
            max_memory_size=2000):
        super().__init__()
        self.env:ChessEnv = env
        self.state_size = int(np.prod(env.observation_space.shape))
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon_greedy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_nn_model()
        self.model.to(self.device)
        self.random_moves = 0
        self.total_moves = 0

    def _build_nn_model(self):
        self.model = nn.Sequential(nn.Linear(self.state_size, 256),
                          nn.ReLU(),
                          nn.Linear(256, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, self.action_size))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
                                self.model.parameters(), self.lr)

    def choose_action(self, state, mask=None):
        if mask is None:
            mask = self.env.get_action_mask()
        self.total_moves += 1
        if np.random.rand() <= self.epsilon:
            legal_actions = np.where(mask == 1)[0]
            self.random_moves += 1
            return np.random.choice(legal_actions)
        

        with torch.no_grad():
            q_values = self.model(state.to(self.device))[0].cpu().numpy()
            q_values[mask == 0] = -np.inf
        return int(np.argmax(q_values))
    
    def remember(self, transition):
        self.memory.append(transition)

    def _learn(self, batch_samples):
        batch_states, batch_targets = [], []
        for transition in batch_samples:
            s, a, r, next_s, done = transition
            with torch.no_grad():
                if done:
                    target = r
                else:
                    pred = self.model(next_s.to(self.device))[0]
                    target = r + self.gamma * pred.max()
            target_all = self.model(s)[0]
            target_all[a] = target
            batch_states.append(s.view(-1))
            batch_targets.append(target_all)
        batch_states = torch.stack(batch_states).float().to(self.device)
        batch_targets = torch.stack(batch_targets).float().to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(batch_states)
        loss = self.loss_fn(pred, batch_targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _adjust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        loss = self._learn(samples)
        self._adjust_epsilon()
        return loss

    def save_model(self, path="alpha_chess.pth"):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, path)

    def load_model(self, path="alpha_chess.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]

def reinforcement_training(num_episodes=200, init_replay_memory_size=500, batch_size=32, load_model=True, phase=3):
    print("starting training")
    env = ChessEnv(phase=phase)
    agent = AlphaZeroNet(env)
    enemy_agent = AlphaZeroNet(env)
    if load_model:
        agent.load_model()
    state = env.reset()
    state = state.flatten().unsqueeze(0)
    print("filling buffer")
    for i in range(init_replay_memory_size):
        mask = env.get_action_mask()
        action = agent.choose_action(state, mask)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten().unsqueeze(0)
        agent.remember((state.to(agent.device), action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state.flatten().unsqueeze(0)
    print("buffer filled")
    total_rewards, losses, moves_made, paths = [], [], [], []
    agent1_starts = True
    agent1_turn = True
    print("starting self play")
    for e in range(num_episodes):
        state = env.reset()
        state = state.flatten().unsqueeze(0)
        for i in range(500):
            cur_agent = agent if agent1_turn else enemy_agent
            mask = env.get_action_mask()
            action = cur_agent.choose_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.flatten().unsqueeze(0)
            if agent1_turn:
                cur_agent.remember((state.to(cur_agent.device), action, reward, next_state, done))
                total_rewards.append(reward)
            if len(agent.memory) > batch_size and i % 5 == 0:
                loss = agent.replay(batch_size)
                losses.append(loss)
            if e % 100 == 0:
                env.render()
            if done:
                moves_made.append(i+1)
                break
            agent1_turn = not agent1_turn
        agent1_starts = not agent1_starts
        agent1_turn = agent1_starts
        if (e+1) % 10 == 0:
            print(f"Episodes: {e+1}/{num_episodes}, Total Reward: {np.sum(total_rewards)}, avg # moves: {np.sum(moves_made)/len(moves_made):0.2f}")
            total_rewards = []
            moves_made = []
        if e % 1000 == 0:
            new_path = "Alpha_Chess" + str(e//1000) + ".pth"
            agent.save_model(new_path)
            paths.append(new_path)
            if len(paths) > 3:
                deleted_path = paths.pop(0)
                os.remove(deleted_path)
            rand_path = random.choice(paths)
            enemy_agent.load_model(rand_path)
    env.close()


def timed_input(prompt, timeout=5):
    user_input = [None]

    def get_input():
        try:
            user_input[0] = input(prompt)
        except EOFError:
            pass

    thread = threading.Thread(target=get_input)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("\n⏳ No response. Continuing training automatically...")
        return None
    return user_input[0]

def main():
    reinforcement_training(num_episodes=10000, load_model=False, phase=1)


if __name__ == "__main__":
    main()

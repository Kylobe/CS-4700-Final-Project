import torch
import torch.nn as nn
import torch.nn.functional as F
from ChessEnv import ChessEnv
import random
from collections import deque
import numpy as np
import os

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class CNNAlphaZeroNet(nn.Module):
    def __init__(self, env:ChessEnv, num_res_blocks=6, num_channels=128, learning_rate=5e-5, weight_decay=1e-5):
        super().__init__()
        self.env:ChessEnv = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=20000)

        # Initial convolutional layer
        self.conv_in = nn.Sequential(
            nn.Conv2d(18, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Residual tower
        self.res_tower = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, self.action_size)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()   # outputs value in range [-1,1]
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.SmoothL1Loss()  # More stable than MSE for noisy rewards
        self.to(self.device)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def predict(self, state):
        """Forward pass returning policy logits and value prediction."""
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(state.to(self.device))
        return policy, value

    def choose_action(self, state, mask=None, epsilon=0.1):
        """Epsilon-greedy action selection with legal action masking."""
        if mask is None:
            mask = self.env.get_action_mask()

        if torch.rand(1).item() < epsilon:
            legal_actions = torch.where(torch.tensor(mask, device=self.device) == 1)[0]
            return int(legal_actions[torch.randint(0, len(legal_actions), (1,))])

        policy_logits, _ = self.forward(state.to(self.device))
        policy = policy_logits.squeeze().detach().cpu().numpy()
        policy[mask == 0] = -1e9
        return int(policy.argmax())

    def save_model(self, path="alpha_chess_cnn.pth"):
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_model(self, path="alpha_chess_cnn.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.to(self.device)

    def _learn(self, batch_samples, gamma=0.99, value_loss_weight=0.5, clip_grad=1.0):
        """
        batch_samples: list of (state, action, reward, next_state, done)
        gamma: discount factor
        value_loss_weight: weight for value loss in total loss
        """
        # Prepare tensors
        states = torch.cat([s for s, _, _, _, _ in batch_samples]).to(self.device)
        actions = torch.tensor([a for _, a, _, _, _ in batch_samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in batch_samples], dtype=torch.float32, device=self.device)
        next_states = torch.cat([ns for _, _, _, ns, _ in batch_samples]).to(self.device)
        dones = torch.tensor([d for _, _, _, _, d in batch_samples], dtype=torch.float32, device=self.device)

        # --------------------
        # Forward pass current states
        # --------------------
        policy_logits, state_values = self.forward(states)
        q_values = policy_logits.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --------------------
        # Compute TD targets
        # --------------------
        with torch.no_grad():
            next_policy_logits, _ = self.forward(next_states)
            next_q_values = next_policy_logits.max(1)[0]
            td_targets = rewards + gamma * next_q_values * (1 - dones)
            td_targets = torch.clamp(td_targets, -1.0, 1.0)


        # --------------------
        # Loss functions
        # --------------------
        policy_loss = F.smooth_l1_loss(q_values, td_targets)

        # Value head tries to predict long-term returns
        value_loss = F.mse_loss(state_values.squeeze(), td_targets)

        # Combine losses (AlphaZero style)
        loss = policy_loss + value_loss_weight * value_loss

        # --------------------
        # Backpropagation
        # --------------------
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

        self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def replay(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        loss, pol_loss, val_loss = self._learn(samples)
        return loss, pol_loss, val_loss

    def remember(self, transition):
        # Automatically move states to CPU to save GPU memory
        s, a, r, ns, d = transition
        self.memory.append((s.cpu(), a, r, ns.cpu(), d))



def reinforcement_training(num_episodes=200, init_replay_memory_size=500, batch_size=32, load_model=True, phase=3):
    print("starting training")
    env = ChessEnv(phase=phase)
    agent = CNNAlphaZeroNet(env)
    enemy_agent = CNNAlphaZeroNet(env)
    if load_model:
        agent.load_model("CNNAlpha_Chess.pth")
        enemy_agent.load_model("Alpha_chess399.pth")
    state = env.reset()
    state = state.unsqueeze(0)
    print("filling buffer")
    for i in range(init_replay_memory_size):
        mask = env.get_action_mask()
        action = agent.choose_action(state, mask)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.unsqueeze(0)
        agent.remember((state, action, np.tanh(reward/100), next_state, done))
        if done:
            state = env.reset()
            state = state.unsqueeze(0)
    print("buffer filled")
    total_rewards, losses, moves_made, paths = [], [], [], []
    agent1_starts = True
    agent1_turn = True
    print("starting self play")
    for e in range(num_episodes):
        state = env.reset()
        state = state.unsqueeze(0)
        for i in range(500):
            cur_agent = agent if agent1_turn else enemy_agent
            mask = env.get_action_mask()
            action = cur_agent.choose_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.unsqueeze(0)
            if agent1_turn:
                agent.remember((state, action, np.tanh(reward/100), next_state, done))
                total_rewards.append(reward)
            if len(agent.memory) > batch_size:
                loss, pol_loss, val_loss = agent.replay(batch_size)
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
            print(f"Episodes: {e+1}/{num_episodes}, "
                f"Avg Reward: {np.mean(total_rewards):0.2f}, "
                f"Moves: {np.mean(moves_made):.02f}, "
            )
            loss_val = np.mean(losses) if losses else 0.0
            print(f"Loss: {loss_val:0.4f}")
            total_rewards = []
            moves_made = []
        if e % 50 == 0:
            new_path = "Alpha_Chess" + str(e//50) + ".pth"
            agent.save_model(new_path)
            paths.append(new_path)
            if len(paths) > 3:
                deleted_path = paths.pop(0)
                os.remove(deleted_path)
            rand_path = random.choice(paths)
            enemy_agent.load_model(rand_path)
    agent.save_model("CNNAlpha_Chess.pth")
    env.close()


def main():
    reinforcement_training(num_episodes=200000, load_model=True, phase=1)


if __name__ == "__main__":
    main()

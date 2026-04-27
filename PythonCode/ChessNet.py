import torch
import torch.nn as nn
import torch.nn.functional as F
from ChessEnv import ChessEnv
import chess
import numpy as np
import random
from multiprocessing import Pool

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.startBlock = nn.Sequential(
            nn.Conv2d(17, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True),
        )

        self.backBone = nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resBlocks)])

        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hidden, 73, kernel_size=1, padding=0, bias=True),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8*8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x):
        x = self.startBlock(x)
        for b in self.backBone:
            x = b(x)

        policy_planes = self.policy_conv(x)
        value = self.value_head(x)
        return policy_planes, value

class ChessNetTrainer:
    def __init__(self, model, optimizer, env, args):
        self.model:ChessNet = model
        self.optimizer = optimizer
        self.env:ChessEnv = env
        self.args = args

    def train_loader(self, loader):
        self.model.train()
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        batches = 0

        for batch in loader:
            if isinstance(batch, dict):
                states = batch["state"]
                policy_targets = batch["policy"]
                value_targets = batch["value"]
                action_mask = batch.get("action_mask", None)
            else:
                if len(batch) == 4:
                    states, policy_targets, value_targets, action_mask = batch
                elif len(batch) == 3:
                    states, policy_targets, value_targets = batch
                    action_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure length={len(batch)}")

            states = states.float().to(self.model.device, non_blocking=True)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device).view(-1, 1)

            if not torch.is_tensor(policy_targets):
                policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32)
            policy_targets = policy_targets.to(self.model.device, non_blocking=True)

            if action_mask is not None:
                if not torch.is_tensor(action_mask):
                    action_mask = torch.tensor(np.array(action_mask), dtype=torch.float32)
                action_mask = action_mask.to(self.model.device, non_blocking=True)

            out_policy, out_value = self.model(states)

            if action_mask is not None:
                NEG = -1e9
                masked_logits = out_policy.masked_fill(action_mask == 0, NEG)  # [B,73,8,8]
                log_probs = F.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
                policy_loss = -(policy_targets * log_probs).sum(dim=(1,2,3)).mean()
            else:
                log_probs = F.log_softmax(out_policy.flatten(1), dim=1)
                policy_loss = -(policy_targets.flatten(1) * log_probs).sum(dim=1).mean()

            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            batches += 1

        if batches == 0:
            return 0.0, 0.0, 0.0

        return total_loss / batches, total_policy / batches, total_value / batches

    def train(self, memory):
        random.shuffle(memory)
        total_losses = []
        policy_losses = []
        value_losses = []

        for batch_indx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batch_indx: batch_indx + self.args["batch_size"]]
            if not sample:
                continue

            state, policy_targets, value_targets, action_mask = zip(*sample)
            state = torch.stack(state).float().to(self.model.device)

            policy_targets = torch.tensor(
                np.array(policy_targets), dtype=torch.float32, device=self.model.device
            )

            value_targets = torch.tensor(
                np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device
            )

            action_mask = torch.tensor(
                np.array(action_mask),
                dtype=torch.float32,
                device=self.model.device
            )

            out_policy, out_value = self.model(state)

            p = policy_targets.clamp(min=1e-12)
            entropy = -(p * torch.log(p)).sum(dim=(1, 2, 3))

            NEG = -1e9
            masked_logits = out_policy.masked_fill(action_mask == 0, NEG)

            log_probs = F.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
            policy_loss = -(policy_targets * log_probs).sum(dim=(1,2,3)).mean()

            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_losses.append(loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return (
            sum(total_losses) / len(total_losses),
            sum(policy_losses) / len(policy_losses),
            sum(value_losses) / len(value_losses),
        )




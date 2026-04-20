import argparse
import sqlite3
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from AlphaZeroChess import AlphaZeroChess
from ChessEnv import ChessEnv
from stockfish_data_gen import count_db_samples, open_and_init_db, unpack_sparse_policy


class SQLitePolicyValueDataset(Dataset):
    def __init__(self, db_path: str, validation: bool):
        self.db_path = str(Path(db_path).resolve())
        self.validation = validation
        self._conn = None

        with open_and_init_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT k FROM labels WHERE is_validation_position = ? AND fen IS NOT NULL ORDER BY k",
                (int(validation),)
            ).fetchall()
        self.keys = [row[0] for row in rows]

    def __len__(self):
        return len(self.keys)

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
        return self._conn

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        row = self._get_conn().execute(
            "SELECT fen, value, policy_idx, policy_prob FROM labels WHERE k=?",
            (key,),
        ).fetchone()
        if row is None:
            raise IndexError(f"Missing row for key {key}")

        fen, value, idx_blob, prob_blob = row
        board = chess.Board(fen=fen)
        state = ChessEnv.encode_board(board).float()
        action_mask = torch.tensor(ChessEnv.create_plane_action_mask(board), dtype=torch.float32)
        policy = torch.zeros_like(action_mask)
        policy_idx, policy_prob = unpack_sparse_policy(idx_blob, prob_blob)
        if len(policy_idx) > 0:
            policy.view(-1)[torch.from_numpy(np.asarray(policy_idx, dtype=np.int64))] = torch.from_numpy(
                np.asarray(policy_prob, dtype=np.float32)
            )
        value = torch.tensor(float(value), dtype=torch.float32)
        return state, policy, value, action_mask

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()


def compute_policy_loss(policy_logits, policy_targets, action_mask):
    neg = -1e9
    masked_logits = policy_logits.masked_fill(action_mask == 0, neg)
    log_probs = F.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
    return -(policy_targets * log_probs).sum(dim=(1, 2, 3)).mean()


def train_one_epoch(model, loader, optimizer, device, epoch: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False, dynamic_ncols=True)
    for states, policies, values, action_mask in progress:
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).view(-1, 1)
        action_mask = action_mask.to(device, non_blocking=True)

        pred_policy, pred_values = model(states)
        policy_loss = compute_policy_loss(pred_policy, policies, action_mask)
        value_loss = F.mse_loss(pred_values, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        batches += 1
        progress.set_postfix(
            total=f"{total_loss / batches:.4f}",
            policy=f"{total_policy_loss / batches:.4f}",
            value=f"{total_value_loss / batches:.4f}",
        )

    progress.close()

    batches = max(batches, 1)
    return (
        total_loss / batches,
        total_policy_loss / batches,
        total_value_loss / batches,
    )


@torch.no_grad()
def evaluate(model, loader, device, epoch: int, total_epochs: int):
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_mse = 0.0
    total_mae = 0.0
    batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [val]", leave=False, dynamic_ncols=True)
    for states, policies, values, action_mask in progress:
        states = states.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).view(-1, 1)
        action_mask = action_mask.to(device, non_blocking=True)

        pred_policy, pred_values = model(states)
        policy_loss = compute_policy_loss(pred_policy, policies, action_mask)
        mse = F.mse_loss(pred_values, values)
        mae = torch.mean(torch.abs(pred_values - values))
        loss = policy_loss + mse

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_mse += mse.item()
        total_mae += mae.item()
        batches += 1
        progress.set_postfix(
            total=f"{total_loss / batches:.4f}",
            policy=f"{total_policy_loss / batches:.4f}",
            value=f"{total_value_mse / batches:.4f}",
            mae=f"{total_mae / batches:.4f}",
        )

    progress.close()

    if batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_loss / batches,
        total_policy_loss / batches,
        total_value_mse / batches,
        total_mae / batches,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ResNet policy and value heads from stockfish_label_cache.db")
    parser.add_argument("--db-path", default="stockfish_label_cache.db")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-res-blocks", type=int, default=20)
    parser.add_argument("--num-hidden", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=4)
    parser.add_argument("--save-path", default="resnet_policy_value_from_db_best.pt")
    parser.add_argument("--latest-path", default="resnet_policy_value_from_db_latest.pt")
    parser.add_argument("--load-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SQLitePolicyValueDataset(args.db_path, validation=False)
    val_dataset = SQLitePolicyValueDataset(args.db_path, validation=True)

    if len(train_dataset) == 0:
        raise RuntimeError("No non-validation rows with FEN found in the database.")
    if len(val_dataset) == 0:
        raise RuntimeError("No validation rows with FEN found in the database.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = AlphaZeroChess(
        num_resBlocks=args.num_res_blocks,
        num_hidden=args.num_hidden,
    )
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.load_path:
        state_dict = torch.load(args.load_path, map_location=device)
        model.load_state_dict(state_dict)

    best_val_total = float("inf")

    print(f"Starting Training")

    try:
        for epoch in range(1, args.epochs + 1):
            started_at = time.time()
            train_total, train_policy, train_value = train_one_epoch(
                model, train_loader, optimizer, device, epoch, args.epochs
            )
            val_total, val_policy, val_value, val_mae = evaluate(
                model, val_loader, device, epoch, args.epochs
            )
            elapsed = time.time() - started_at

            torch.save(model.state_dict(), args.latest_path)
            if val_total < best_val_total:
                best_val_total = val_total
                torch.save(model.state_dict(), args.save_path)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_total={train_total:.6f} | "
                f"train_policy={train_policy:.6f} | "
                f"train_value={train_value:.6f} | "
                f"val_total={val_total:.6f} | "
                f"val_policy={val_policy:.6f} | "
                f"val_value={val_value:.6f} | "
                f"val_mae={val_mae:.6f} | "
                f"time={elapsed:.1f}s"
            )
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    main()

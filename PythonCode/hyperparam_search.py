import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import optim

from AlphaZeroChess import AlphaZero, AlphaZeroChess
from ChessEnv import ChessEnv
from data_stream import make_stream_loader
from train_value import (
    compute_policy_validation_loss,
    compute_value_validation_loss,
    load_all_data,
)


DEFAULT_SEARCH_SPACE = {
    "lr": [3e-4, 1e-4, 5e-5],
    "weight_decay": [1e-3, 1e-4, 1e-5],
    "res_blocks": [10, 20, 30, 40],
    "num_hidden": [128, 256, 384],
    "batch_size": [64, 128, 256],
    "epochs": [2, 3, 4],
}


@dataclass
class TrialResult:
    trial_index: int
    score: float
    policy_loss: float
    value_loss: float
    train_loss: float
    train_policy_loss: float
    train_value_loss: float
    best_epoch: int
    epochs_ran: int
    elapsed_seconds: float
    config: Dict[str, Any]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_search_space(path: Optional[str]) -> Dict[str, List[Any]]:
    if path is None:
        return DEFAULT_SEARCH_SPACE

    with open(path, "r", encoding="utf-8") as f:
        search_space = json.load(f)

    if not isinstance(search_space, dict):
        raise ValueError("Search space file must contain a JSON object.")

    normalized = {}
    for key, values in search_space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Search space entry '{key}' must be a non-empty list.")
        normalized[key] = values
    return normalized


def sample_config(
    rng: random.Random,
    search_space: Dict[str, List[Any]],
    fixed_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = {key: rng.choice(values) for key, values in search_space.items()}
    if fixed_overrides:
        config.update(fixed_overrides)
    return config


def build_loader(root_dir: str, batch_size: int, seed: int, num_workers: int):
    return make_stream_loader(
        root_dir=resolve_path(root_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=50_000,
        seed=seed,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=4,
        drop_errors=True,
    )


def train_loader_limited(alpha: AlphaZero, loader, max_batches: Optional[int]) -> Tuple[float, float, float]:
    alpha.model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    batches = 0

    for batch in loader:
        if isinstance(batch, dict):
            states = batch["state"]
            policy_targets = batch["policy"]
            value_targets = batch["value"]
            action_mask = batch.get("action_mask")
        else:
            if len(batch) == 4:
                states, policy_targets, value_targets, action_mask = batch
            elif len(batch) == 3:
                states, policy_targets, value_targets = batch
                action_mask = None
            else:
                raise ValueError(f"Unexpected batch structure length={len(batch)}")

        states = states.float().to(alpha.model.device, non_blocking=True)
        value_targets = torch.as_tensor(
            value_targets, dtype=torch.float32, device=alpha.model.device
        ).view(-1, 1)
        policy_targets = torch.as_tensor(
            np.asarray(policy_targets), dtype=torch.float32, device=alpha.model.device
        )

        if action_mask is not None:
            action_mask = torch.as_tensor(
                np.asarray(action_mask), dtype=torch.float32, device=alpha.model.device
            )

        out_policy, out_value = alpha.model(states)

        if action_mask is not None:
            masked_logits = out_policy.masked_fill(action_mask == 0, -1e9)
            log_probs = torch.nn.functional.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
            policy_loss = -(policy_targets * log_probs).sum(dim=(1, 2, 3)).mean()
        else:
            log_probs = torch.nn.functional.log_softmax(out_policy.flatten(1), dim=1)
            policy_loss = -(policy_targets.flatten(1) * log_probs).sum(dim=1).mean()

        value_loss = torch.nn.functional.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss

        alpha.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        alpha.optimizer.step()

        total_loss += loss.item()
        total_policy += policy_loss.item()
        total_value += value_loss.item()
        batches += 1

        if max_batches is not None and batches >= max_batches:
            break

    if batches == 0:
        raise RuntimeError("Training loader produced zero batches. Check the training data directory.")

    return total_loss / batches, total_policy / batches, total_value / batches


def evaluate_trial(
    config: Dict[str, Any],
    train_dir: str,
    val_samples: List[Any],
    max_batches_per_epoch: Optional[int],
    num_workers: int,
    seed: int,
    patience: int,
) -> Tuple[TrialResult, AlphaZeroChess, optim.Optimizer]:
    set_seed(seed)
    env = ChessEnv()
    model = AlphaZeroChess(
        num_resBlocks=int(config["res_blocks"]),
        num_hidden=int(config["num_hidden"]),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    alpha = AlphaZero(model, optimizer, env, config)

    best_score = math.inf
    best_policy_loss = math.inf
    best_value_loss = math.inf
    best_epoch = 0
    epochs_ran = 0
    epochs_without_improvement = 0
    latest_train_loss = 0.0
    latest_train_policy = 0.0
    latest_train_value = 0.0

    start = time.perf_counter()

    for epoch in range(int(config["epochs"])):
        epochs_ran = epoch + 1
        ds, loader = build_loader(train_dir, int(config["batch_size"]), seed + epoch, num_workers)
        ds.set_epoch(epoch)

        latest_train_loss, latest_train_policy, latest_train_value = train_loader_limited(
            alpha=alpha,
            loader=loader,
            max_batches=max_batches_per_epoch,
        )

        policy_val_loss = compute_policy_validation_loss(model, val_samples, int(config["batch_size"]))
        value_val_loss = compute_value_validation_loss(model, val_samples, int(config["batch_size"]))
        total_score = policy_val_loss + value_val_loss

        print(
            f"[epoch {epoch + 1}/{config['epochs']}] "
            f"train_total={latest_train_loss:.4f} "
            f"train_policy={latest_train_policy:.4f} "
            f"train_value={latest_train_value:.4f} "
            f"val_total={total_score:.4f} "
            f"val_policy={policy_val_loss:.4f} "
            f"val_value={value_val_loss:.4f}"
        )

        if total_score + 1e-6 < best_score:
            best_score = total_score
            best_policy_loss = policy_val_loss
            best_value_loss = value_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience >= 0 and epochs_without_improvement > patience:
            break

    elapsed = time.perf_counter() - start

    result = TrialResult(
        trial_index=-1,
        score=best_score,
        policy_loss=best_policy_loss,
        value_loss=best_value_loss,
        train_loss=latest_train_loss,
        train_policy_loss=latest_train_policy,
        train_value_loss=latest_train_value,
        best_epoch=best_epoch,
        epochs_ran=epochs_ran,
        elapsed_seconds=elapsed,
        config=dict(config),
    )
    return result, model, optimizer


def load_completed_trials(results_path: str) -> List[Dict[str, Any]]:
    results_path = resolve_path(results_path)
    if not os.path.exists(results_path):
        return []

    completed = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            completed.append(json.loads(line))
    return completed


def append_trial(results_path: str, result: TrialResult) -> None:
    with open(resolve_path(results_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def save_best_checkpoint(
    output_dir: str,
    result: TrialResult,
    model: AlphaZeroChess,
    optimizer: optim.Optimizer,
) -> None:
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "best_optimizer.pt"))
    with open(os.path.join(output_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search for AlphaZeroChess.")
    parser.add_argument("--train-dir", default="Stockfish_data")
    parser.add_argument("--val-dir", default="Stockfish_test_data")
    parser.add_argument("--results-path", default="hyperparam_search_results.jsonl")
    parser.add_argument("--output-dir", default="hyperparam_search")
    parser.add_argument("--search-space", default=None)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--max-val-samples", type=int, default=50000)
    parser.add_argument("--max-batches-per-epoch", type=int, default=400)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_space = load_search_space(args.search_space)
    os.makedirs(resolve_path(args.output_dir), exist_ok=True)

    print(f"Loading validation data from {args.val_dir}...")
    val_samples = load_all_data(resolve_path(args.val_dir), max_len=args.max_val_samples)
    if not val_samples:
        raise RuntimeError("Validation directory did not yield any .pt samples.")

    completed_trials = load_completed_trials(args.results_path) if args.resume else []
    best_existing_score = min((trial["score"] for trial in completed_trials), default=math.inf)
    rng = random.Random(args.seed + len(completed_trials))

    print(
        f"Starting search with {len(val_samples)} validation samples, "
        f"{len(completed_trials)} completed trials, "
        f"targeting {args.trials} total trials."
    )

    for trial_index in range(len(completed_trials), args.trials + len(completed_trials)):
        config = sample_config(rng, search_space)
        print(f"\nTrial {trial_index + 1}/{args.trials + len(completed_trials)}")
        print(json.dumps(config, sort_keys=True))

        trial_seed = args.seed + trial_index
        result, model, optimizer = evaluate_trial(
            config=config,
            train_dir=args.train_dir,
            val_samples=val_samples,
            max_batches_per_epoch=args.max_batches_per_epoch,
            num_workers=args.loader_workers,
            seed=trial_seed,
            patience=args.patience,
        )
        result.trial_index = trial_index
        append_trial(args.results_path, result)

        print(
            f"trial_score={result.score:.4f} "
            f"policy={result.policy_loss:.4f} "
            f"value={result.value_loss:.4f} "
            f"best_epoch={result.best_epoch} "
            f"time={result.elapsed_seconds:.1f}s"
        )

        if result.score < best_existing_score:
            best_existing_score = result.score
            save_best_checkpoint(args.output_dir, result, model, optimizer)
            print(f"Saved new best checkpoint to {args.output_dir}")

    all_trials = load_completed_trials(args.results_path)
    if not all_trials:
        raise RuntimeError("No completed trials were recorded.")

    best_trial = min(all_trials, key=lambda trial: trial["score"])
    print("\nBest trial:")
    print(json.dumps(best_trial, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

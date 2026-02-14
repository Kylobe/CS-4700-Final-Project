import chess
import chess.pgn
import torch
import random
from typing import List, Tuple, Optional, Iterable
from ChessEnv import ChessEnv  # your env
from AlphaZeroChess import AlphaZeroChess, AlphaZero
from torch import optim
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
import chess.engine
from collections import defaultdict
import itertools
import time
import numbers
import json
import os
from data_stream import make_stream_loader
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

def analyze_value_distribution(samples):
    vals = np.array([v for (_, v) in samples])
    print("Total samples:", len(vals))
    print("Mean:", vals.mean(), "Std:", vals.std())

    # Buckets by |value|
    bins = [0, 0.1, 0.3, 0.6, 1.1]
    labels = ["|v| < 0.1", "0.1–0.3", "0.3–0.6", "> 0.6"]

    for (lo, hi), name in zip(zip(bins[:-1], bins[1:]), labels):
        mask = (np.abs(vals) >= lo) & (np.abs(vals) < hi)
        print(f"{name}: {mask.mean()*100:.1f}% of samples")

def bucket_by_eval(samples):
    buckets = defaultdict(list)
    for state, policy, val, action_mask in samples:
        a = abs(val)
        if a < 0.1:
            buckets["small"].append((state, policy, val, action_mask))
        elif a < 0.3:
            buckets["medium"].append((state, policy, val, action_mask))
        elif a < 0.6:
            buckets["large"].append((state, policy, val, action_mask))
        else:
            buckets["huge"].append((state, policy, val, action_mask))
    return buckets

def build_balanced_dataset(samples, total_size=50000):
    buckets = bucket_by_eval(samples)

    # Decide proportions (tune these)
    proportions = {
        "small": 0.15,
        "medium": 0.25,
        "large": 0.30,
        "huge": 0.30,
    }

    balanced = []
    for name, frac in proportions.items():
        bucket = buckets[name]
        if not bucket:
            continue
        k = min(len(bucket), int(total_size * frac))
        balanced.extend(random.sample(bucket, k))

    random.shuffle(balanced)
    return balanced

def compute_zero_baseline(val_samples):
    # Extract ground-truth value labels
    targets = np.array([v for (_, v) in val_samples], dtype=np.float32)

    # Predictions are all zero
    preds = 0.0

    # MSE loss for constant prediction 0.0
    baseline_loss = np.mean((preds - targets)**2)

    return baseline_loss

def compute_value_validation_loss(model, val_samples, batch_size=128):
    model.eval()  # important for any layers with running stats
    losses = []

    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i : i+batch_size]

            if not batch:
                continue

            states, policy_targets, value_targets, action_mask = zip(*batch)

            # Stack states into tensor
            states = torch.stack(states).float().to(model.device)

            # Convert value targets
            value_targets = torch.tensor(value_targets, dtype=torch.float32,
                                         device=model.device).view(-1, 1)

            # Forward pass ONLY
            _, out_value = model(states)

            loss = F.mse_loss(out_value, value_targets)
            losses.append(loss.item())

    if len(losses) == 0:
        return 0.0

    return sum(losses) / len(losses)

def compute_policy_validation_loss(model, val_samples, batch_size=128):
    """
    Computes cross-entropy loss of the policy head using test samples.

    val_samples: list of (state_tensor, policy_target)
                 - state_tensor: torch.Tensor [C,8,8]
                 - policy_target: numpy array or list of floats, shape = (num_moves,)
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i : i+batch_size]
            if not batch:
                continue

            states, policy_targets, value_targets, action_mask = zip(*batch)

            # Stack states -> shape: [B, C, 8, 8]
            states = torch.stack(states).float().to(model.device)

            # Convert policy targets -> shape: [B, num_moves]
            policy_targets = torch.tensor(
                np.array(policy_targets, dtype=np.float32),
                dtype=torch.float32,
                device=model.device
            )

            action_mask = torch.tensor(
                np.array(action_mask),
                dtype=torch.float32,
                device=model.device
            )

            # Forward pass
            out_policy, _ = model(states)

            NEG = -1e9
            masked_logits = out_policy.masked_fill(action_mask == 0, NEG)  # [B,73,8,8]

            log_probs = F.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
            policy_loss = -(policy_targets * log_probs).sum(dim=(1,2,3)).mean()

            losses.append(policy_loss.item())

    return sum(losses) / len(losses) if losses else 0.0


def result_to_winner(result_str: str) -> Optional[bool]:
    """
    Convert PGN 'Result' to winner color:
    - '1-0'  -> chess.WHITE
    - '0-1'  -> chess.BLACK
    - '1/2-1/2' or others -> None (draw or unknown)
    """
    if result_str == "1-0":
        return chess.WHITE
    elif result_str == "0-1":
        return chess.BLACK
    else:  # '1/2-1/2', '*', etc.
        return None


def value_from_pov(side_to_move: bool, winner: Optional[bool]) -> float:
    """
    Return value label from side-to-move POV:
    - +1 if side_to_move eventually wins
    - -1 if side_to_move eventually loses
    -  0 if draw or unknown
    """
    if winner is None:
        return 0.0
    return 1.0 if side_to_move == winner else -1.0

def eval_to_value(score: chess.engine.PovScore, turn: bool) -> float:
    """
    Convert Stockfish evaluation to a value in [-1, 1].
    Turn = side to move (True = white, False = black).
    """

    # Get the score from the POV of side to move
    sc = score.white() if turn == chess.WHITE else score.black()

    # Case 1: Mate
    if sc.is_mate():
        mate_in = sc.mate()  # positive = winning, negative = losing
        if mate_in > 0:
            return 1.0
        else:
            return -1.0

    # Case 2: Centipawns
    cp = sc.score()  # centipawns

    # Clip large evals (avoids extreme values)
    cp = max(min(cp, 1000), -1000)

    # Scale to [-1, 1]
    return cp / 1000.0


def parse_pgn_files_for_value_and_policy(
    pgn_paths: Iterable[str],
    env: ChessEnv,
    engine,
    max_games: Optional[int] = None,
    max_positions_per_game: Optional[int] = None,
    shuffle_games: bool = True,
    use_policy: bool = True,
) -> List[Tuple[torch.Tensor, int, float]]:
    """
    Parse one or more PGN files and return a list of training samples.

    If use_policy=True:
        Each sample is (state_tensor, policy_idx, value_label)
    If use_policy=False:
        Each sample is (state_tensor, value_label)   # policy_idx is omitted

    - state_tensor: output of ChessEnv._encode_board(board) [C x 8 x 8]
    - policy_idx  : integer index produced by env.encode_action(move, board.turn)
    - value_label : float in {-1.0, 0.0, +1.0} from side-to-move POV
    """
    samples = []
    
    paths = list(pgn_paths)
    if shuffle_games:
        random.shuffle(paths)

    games_processed = 0
    
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break  # end of this file

                # Respect max_games limit
                games_processed += 1
                if max_games is not None and games_processed > max_games:
                    return samples


                # Skip completely unfinished games ('*') if you want strictly labeled data
                # (You *can* keep them as draw=0 if you want, but safest is to skip.)

                board = game.board()

                positions_added = 0

                # Iterate through the mainline (no variations for now)
                for move in game.mainline_moves():
                    # Limit number of positions per game (optional)
                    if max_positions_per_game is not None and positions_added >= max_positions_per_game:
                        break

                    side_to_move = board.turn
                    info = engine.analyse(board, chess.engine.Limit(depth=10))
                    value_label = eval_to_value(info["score"], board.turn)
                    # Encode board from side-to-move perspective
                    state_tensor = ChessEnv._encode_board(board)  # torch.Tensor

                    if use_policy:
                        try:
                            policy_idx = env.encode_action(move, side_to_move)
                        except KeyError:
                            # Move not in your move vocabulary → skip this position
                            board.push(move)
                            continue

                        samples.append((state_tensor, policy_idx, value_label))
                    else:
                        samples.append((state_tensor, value_label))

                    positions_added += 1
                    if len(samples) % 1000 == 0:
                        torch.save(samples, "labeled.pt")

                    # Now actually make the move and continue
                    board.push(move)

    torch.save(samples, "labeled.pt")
    return samples

def train_on_hyper_params(args, data, logging = False, early_stopping = True):
    env = ChessEnv()
    model = AlphaZeroChess(env, num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    optimizer = optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    alpha = AlphaZero(model, optimizer, env, args)
    split = int(0.8 * len(data))
    train = data[:split]
    test = data[split:]
    model.train()
    best_validation_loss = np.inf
    completed_epochs = 0
    if logging:
        print(f"Starting training with {args}")
    for epoch in range(args['epochs']):
        completed_epochs += 1
        total_loss, policy_loss, value_loss = alpha.train(train)
        policy_val_loss = compute_policy_validation_loss(model, test, args["batch_size"])
        value_val_loss = compute_value_validation_loss(model, test, args["batch_size"])
        total_validation_loss = policy_val_loss + value_val_loss
        if logging:
            print(f"[Epoch: {epoch + 1}/{args['epochs']}]\tTraining Loss: {total_loss:.2f},\tPolicy Training Loss: {policy_loss:.2f},\tValue Training Loss: {value_loss:.2f},\tValidation Loss: {total_validation_loss:.2f},\tPolicy Validation Loss: {policy_val_loss:.2f},\tValue Validation Loss: {value_val_loss:.2f},\tTime: {time.ctime()}")
        if total_validation_loss - best_validation_loss > 0.05 and early_stopping:
            if logging:
                print(f"Early stopping detected, validation loss not improving.")
            break
        if total_validation_loss < best_validation_loss:
            best_validation_loss = total_validation_loss
    return best_validation_loss, model, optimizer



def hyperparam_search(data):
    search_space = {
        "lr": [1e-3, 5e-4, 1e-4],
        "weight_decay": [1e-3, 5e-4, 1e-4],
        "res_blocks": [6, 12, 18, 24],
        "num_hidden": [64, 128, 256],
        "batch_size": [128, 256, 512],
        "epochs": [1],
    }

    keys = list(search_space.keys())
    best_config = None
    best_metric = float("inf")

    for values in itertools.product(*(search_space[k] for k in keys)):
        config = dict(zip(keys, values))
        print("Testing config:", config)
        metric = train_on_hyper_params(config, data)
        print("  -> metric:", metric)

        if metric < best_metric:
            best_metric = metric
            best_config = config

    print("\nBest config:", best_config)
    print("Best val metric:", best_metric)
    return best_config, best_metric

class Config:
    def __init__(self, config: dict, val: float, search_time):
        self.config = config
        self.val = val
        self.search_time = search_time

    def __lt__(self, other):
        if isinstance(other, Config):
            return self.val < other.val

    def __str__(self):
        def fmt(v):
            if isinstance(v, numbers.Integral):
                return str(v)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return str(v)
            if v != 0 and abs(v) < 0.01:
                return f"{v:.2e}"
            else:
                return f"{v:.2f}"

        parts = []
        for key, value in self.config.items():
            parts.append(f"{key}: {fmt(value)}")

        parts.append(f"val: {fmt(self.val)}")
        parts.append(f"time: {int(self.search_time)}")

        return "(" + ", ".join(parts) + ")"

    def __repr__(self):
        return str(self)

def checkpoint_search(config, val, search_time, path):
    def to_jsonable(x):
        # numpy scalar -> python scalar
        if isinstance(x, np.generic):
            return x.item()
        # numpy arrays -> lists
        if isinstance(x, np.ndarray):
            return x.tolist()
        # dict -> recurse
        if isinstance(x, dict):
            return {k: to_jsonable(v) for k, v in x.items()}
        # list/tuple -> recurse
        if isinstance(x, (list, tuple)):
            return [to_jsonable(v) for v in x]
        return x
    record = dict(config)
    record["val"] = val
    record["time"] = search_time

    record = to_jsonable(record)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def load_checkpoint_search(path):
    if not os.path.exists(path):
        return []

    configs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            configs.append(Config(
                config={k: v for k, v in d.items() if k not in ("val", "time")},
                val=d["val"],
                search_time=d["time"]
            ))
    return configs


def random_hyperparam_search(data, n, k, path = "search_results.jsonl", load = False):
    print(f"Starting Search With {len(data)} Data Points")
    configs = []

    if load:
        configs = load_checkpoint_search(path)
    i = len(configs)
    while i < n:
        start = time.perf_counter()
        cur_config = {
            "lr": 10 ** np.random.uniform(-5, -1),
            "weight_decay": 10 ** np.random.uniform(-5, -1),
            "res_blocks": np.random.choice([6, 12, 18, 24, 30]),
            "num_hidden": np.random.choice([64, 128, 256, 512]),
            "batch_size": np.random.choice([128, 256, 512]),
            "epochs": np.random.randint(3, 16),
        }
        metrics = train_on_hyper_params(cur_config, data)
        metric = metrics[0]
        cur_config["epochs"] = metrics[3]
        end = time.perf_counter()
        search_time = end - start
        configs.append(Config(cur_config, metric, search_time))
        time_str = f" {search_time: 0.2f}s;"
        print_str = f"last run took:"
        if len(time_str) < 8:
            for _ in range(8 - len(time_str)):
                print_str += " "
        print_str += time_str
        checkpoint_search(config=cur_config, val=metric, search_time=search_time, path=path)
        i += 1
        print(print_str + f" {(i * 100) / n: 0.1f}% done")

    configs.sort()
    if len(configs) > k:
        return configs[:k]
    else:
        return configs

def finalize_configs(configs: list[Config], data, k=1, epochs=20, seeds=(0,1,2,3,4)):
    results = []
    for i, cfg in enumerate(configs, 1):
        cfg_dict = dict(cfg.config)
        cfg_dict["epochs"] = epochs

        losses = []
        print(f"Testing config {i + 1} out of {len(configs)}")
        for s in seeds:
            np.random.seed(s)
            loss, *_ = train_on_hyper_params(cfg_dict, data)
            losses.append(loss)
            print(f"\t{((s + 1) * 100) / len(seeds):0.1f}% done")

        mean = float(np.mean(losses))
        std  = float(np.std(losses))
        results.append((mean, std, cfg_dict, losses))

        print(f"[{i}/{len(configs)}] mean={mean:.4f} std={std:.4f} cfg={cfg_dict}")

    results.sort(key=lambda x: x[0])
    top = results[:k]
    return top

def load_all_data(data_dir, max_len = 100000):
    memory = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".pt"):
            path = os.path.join(data_dir, fname)
            chunk = torch.load(path, map_location="cpu", weights_only=False)
            memory.extend(chunk)
        if len(memory) > max_len:
            return memory
    return memory

def main():
    env = ChessEnv()

    config = {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'res_blocks': 40,
        'num_hidden': 256,
        'batch_size': 256,
        'epochs': 50
    }

    model = AlphaZeroChess(env, num_resBlocks=config['res_blocks'], num_hidden=config['num_hidden'])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    alpha = AlphaZero(model, optimizer, env, config)

    # ---- Streaming train loader (ALL 47GB) ----
    stream_ds, train_loader = make_stream_loader(
        root_dir="Stockfish_data",
        batch_size=config["batch_size"],
        num_workers=8,          # tune: 4/8/12 depending on CPU
        shuffle_buffer=50_000,  # tune down if RAM spikes
        seed=123,
        pin_memory=(model.device.type == "cuda"),
    )

    # ---- Small validation set in RAM (e.g., first N samples) ----
    # Easiest: reuse your old loader but cap it
    val_samples = load_all_data("Stockfish_data_val", max_len=50_000) \
        if os.path.isdir("Stockfish_data_val") else load_all_data("Stockfish_data", max_len=50_000)

    best_policy_loss = compute_policy_validation_loss(model, val_samples, config["batch_size"])
    best_value_loss = compute_value_validation_loss(model, val_samples, config["batch_size"])
    bad_epochs = 0
    patience = 2
    print(f"Before Training The Total Validation Loss: {best_policy_loss + best_value_loss}, Policy Loss: {best_policy_loss}, Value Loss: {best_value_loss}")
    print("training (streaming)")
    for epoch in range(config["epochs"]):
        stream_ds.set_epoch(epoch)

        total_loss, policy_loss, value_loss = alpha.train_loader(train_loader)

        scheduler.step()
        policy_val_loss = compute_policy_validation_loss(model, val_samples, config["batch_size"])
        value_val_loss  = compute_value_validation_loss(model, val_samples, config["batch_size"])
        if best_value_loss > value_val_loss:
            best_value_loss = value_val_loss
        elif np.abs(best_value_loss - value_val_loss) > 0.005:
            bad_epochs += 1
        if best_policy_loss > policy_val_loss:
            best_policy_loss = policy_val_loss
        elif np.abs(best_policy_loss - policy_val_loss) > 0.01:
            bad_epochs += 1
        total_val_loss  = policy_val_loss + value_val_loss

        print(f"[Epoch {epoch+1}/{config['epochs']}] "
              f"Train: total={total_loss:.2f} pol={policy_loss:.2f} val={value_loss:.2f} | "
              f"Val: total={total_val_loss:.2f} pol={policy_val_loss:.2f} val={value_val_loss:.2f}")
        if bad_epochs >= patience:
            print(f"Model Validition Loss Is Not Improving. Stopping Training.")
            break

    torch.save(model.state_dict(), "PretrainModel.pt")
    torch.save(optimizer.state_dict(), "PretrainOptimizer.pt")




if __name__ == "__main__":
    main()




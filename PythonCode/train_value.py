import torch
import random
from ChessEnv import ChessEnv
from ChessNet import ChessNet,  ChessNetTrainer
from torch import optim
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import time
import os
from data_stream import make_stream_loader
from stockfish_data_gen import parallel_data_gen, get_engine_path
import multiprocessing as mp
import shutil

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
        balanced.extend(random.choices(bucket, k=k))

    random.shuffle(balanced)
    return balanced

def compute_zero_baseline(val_samples):
    targets = np.array([v for (_, v) in val_samples], dtype=np.float32)

    preds = 0.0

    baseline_loss = np.mean((preds - targets)**2)

    return baseline_loss

def compute_value_validation_loss(model, val_samples, batch_size=128):
    model.eval()
    losses = []

    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i : i+batch_size]

            if not batch:
                continue

            states, policy_targets, value_targets, action_mask = zip(*batch)

            states = torch.stack(states).float().to(model.device)

            value_targets = torch.tensor(value_targets, dtype=torch.float32,
                                         device=model.device).view(-1, 1)

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

            states = torch.stack(states).float().to(model.device)

            policy_targets = torch.stack([
                torch.as_tensor(target, dtype=torch.float32)
                for target in policy_targets
            ]).to(model.device)

            action_mask = torch.stack([
                torch.as_tensor(mask, dtype=torch.float32)
                for mask in action_mask
            ]).to(model.device)

            out_policy, _ = model(states)

            NEG = -1e9
            masked_logits = out_policy.masked_fill(action_mask == 0, NEG)
            log_probs = F.log_softmax(masked_logits.flatten(1), dim=1).view_as(masked_logits)
            policy_loss = -(policy_targets * log_probs).sum(dim=(1,2,3)).mean()

            losses.append(policy_loss.item())

    return sum(losses) / len(losses) if losses else 0.0

def train_on_hyper_params(args, data, logging = False, early_stopping = True):
    env = ChessEnv()
    model = ChessNet(env, num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    optimizer = optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    trainer = ChessNetTrainer(model, optimizer, env, args)
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
        total_loss, policy_loss, value_loss = trainer.train(train)
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
    folder_name = "Stockfish_data"
    validation_folder = "Stockfish_test_data"
    env = ChessEnv()

    config = {
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'res_blocks': 20,
        'num_hidden': 256,
        'batch_size': 64,
        'epochs': 3
    }
    model = ChessNet(num_resBlocks=config['res_blocks'], num_hidden=config['num_hidden'])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    model.load_state_dict(torch.load("CurBestPretrainModel.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    opt_state = torch.load("CurBestOptim.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
    optimizer.load_state_dict(opt_state)
    trainer = ChessNetTrainer(model, optimizer, env, config)
    NUM_WORKERS = 10
    FILES_PER_WORKER = 700
    LENGTH_OF_FILE = 1000
    mill_pos_trained_on = 7_000_000
    print("LR:", optimizer.param_groups[0]["lr"])
    best_policy_loss = 2.22
    best_value_loss = 0.6
    while True:
        val_data = load_all_data(validation_folder, max_len=200_000)
        pre_policy_loss = compute_policy_validation_loss(model, val_data, config["batch_size"])
        pre_value_loss = compute_value_validation_loss(model, val_data, config["batch_size"])
        bad_epochs = 0
        patience = 0
        if best_value_loss > pre_value_loss and best_policy_loss > pre_policy_loss:
            best_value_loss = pre_value_loss
            best_policy_loss = pre_policy_loss

        ds, loader = make_stream_loader(
            root_dir="C:\\Users\\Traedon Harris\\Documents\\GitHub\\CS-4700-Final-Project\\Stockfish_data",
            batch_size=trainer.args["batch_size"],
            num_workers=4,
            shuffle_buffer=50_000,
            seed=0,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4
        )

        print(f"Before Training The Total Validation Loss: {pre_policy_loss + pre_value_loss:.2f}, Policy Loss: {pre_policy_loss:.2f}, Value Loss: {pre_value_loss:.2f}")
        print(f"Best Total Loss: {best_value_loss + best_policy_loss:0.2f}, Best Policy Loss: {best_policy_loss}, Best Value Loss: {best_value_loss}")
        print("training (streaming)")
        for epoch in range(config["epochs"]):
            ds.set_epoch(epoch)

            total_loss, policy_loss, value_loss = trainer.train_loader(loader)

            policy_val_loss = compute_policy_validation_loss(model, val_data, config["batch_size"])
            value_val_loss  = compute_value_validation_loss(model, val_data, config["batch_size"])
            if best_value_loss > value_val_loss and best_policy_loss > policy_val_loss:
                best_value_loss = value_val_loss
                best_policy_loss = policy_val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), "CurBestPretrainModel.pt")
                torch.save(optimizer.state_dict(), "CurBestOptim.pt")
            elif np.abs(best_value_loss - value_val_loss) > 0.005 or np.abs(best_policy_loss - policy_val_loss) > 0.01:
                bad_epochs += 1
            total_val_loss  = policy_val_loss + value_val_loss

            print(f"[Epoch {epoch+1}/{config['epochs']}] "
                f"Train: total={total_loss:.2f} pol={policy_loss:.2f} val={value_loss:.2f} | "
                f"Val: total={total_val_loss:.2f} pol={policy_val_loss:.2f} val={value_val_loss:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']}")
            if bad_epochs > patience:
                print(f"Model Validition Loss Is Not Improving. Stopping Training.")
                break
            model.load_state_dict(torch.load("CurBestPretrainModel.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            opt_state = torch.load("CurBestOptim.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
            optimizer.load_state_dict(opt_state)
        mill_pos_trained_on += NUM_WORKERS * FILES_PER_WORKER * LENGTH_OF_FILE
        print(f"Positions Trained On In Millions: {mill_pos_trained_on / 1_000_000}")

        try:
            shutil.rmtree(folder_name)
            print(f"Directory and all contents deleted: {folder_name}")
        except OSError as e:  
            print(f"Error: {folder_name} : {e.strerror}")
        val_data = []
        try:
            os.makedirs(folder_name, exist_ok=True)
            print(f"Directory recreated: {folder_name}")
        except OSError as e:
            print(f"Error: {folder_name} : {e.strerror}")

        parallel_data_gen(
            num_workers=NUM_WORKERS,
            files_per_worker=FILES_PER_WORKER,
            length_of_file=LENGTH_OF_FILE,
            out_dir=folder_name,
            engine_path=get_engine_path(),
            db_path="stockfish_label_cache.db",
            stagger_seconds=10
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()




import chess
import torch
from ChessEnv import ChessEnv  # your env
import torch
import numpy as np
import chess.engine
from collections import defaultdict
import math
import random as r
import pandas as pd
import os
import multiprocessing as mp
import sqlite3
import json
import time
from typing import Optional, Tuple, List
import chess.polyglot

def sf_scores_to_probs_plane(info_list, board):
    # policy target is plane-based
    policy = np.zeros((73, 8, 8), dtype=np.float32)

    logits = []
    coords = []  # (p, r, c)

    for info in info_list:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None:
            continue

        move = pv[0]

        # NEW: encode to (plane, row, col) where row/col represent FROM square
        try:
            p, r, c = ChessEnv.encode_action(move, board)
        except KeyError:
            continue

        sc = score.white() if board.turn == chess.WHITE else score.black()

        if sc.is_mate():
            mate_in = sc.mate()
            logit = 10.0 if (mate_in is not None and mate_in > 0) else -10.0
        else:
            cp = sc.score()
            if cp is None:
                continue
            cp = max(min(cp, 2000), -2000)
            logit = cp / 200.0

        logits.append(logit)
        coords.append((p, r, c))

    # If nothing found, fallback: uniform over legal moves (plane mask)
    if not logits:
        mask = ChessEnv.create_plane_action_mask(board)  # (73,8,8)
        s = mask.sum()
        if s == 0:
            return policy
        return (mask / s).astype(np.float32)

    T = 0.25  # smaller = sharper (try 0.25, 0.5, 1.0)
    exp_logits = np.exp((logits - np.max(logits)) / T)
    probs = exp_logits / exp_logits.sum()

    for (p, r, c), pr in zip(coords, probs):
        policy[p, r, c] = pr

    # Optional (recommended): renormalize over filled entries (or over legal moves)
    total = policy.sum()
    if total > 0:
        policy /= total

    return policy


def analyze_value_distribution(samples):
    vals = np.array([v for (_, _, v, _) in samples])
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
    for state, policy, val, mask in samples:
        a = abs(val)
        if a < 0.1:
            buckets["small"].append((state, policy, val, mask))
        elif a < 0.3:
            buckets["medium"].append((state, policy, val, mask))
        elif a < 0.6:
            buckets["large"].append((state, policy, val, mask))
        else:
            buckets["huge"].append((state, policy, val, mask))
    return buckets

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
    cp = max(min(cp, 1000), -1000)
    # Scale to [-1, 1]
    return math.tanh(cp / 300)



def data_gen(env: ChessEnv, engine, n = None, length_of_file = 10000):
    memory = []
    files_saved = 0
    while files_saved < n:
        board = env.reset()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if len(legal_moves) <= 0:
                break
            label, action = label_board(engine, board)
            memory.append(label)
            board.push(action)
        if len(memory) % 10 == 0:
            print(f"Memory size is currently: {len(memory)}")
        if len(memory) >= length_of_file:
            files_saved += 1
            torch.save(memory, f"Stockfish_test_data\\labeled{files_saved}.pt")
            print(f"Created new file labeled{files_saved}.pt")
            memory = []

def label_board(engine, board: chess.Board):
    state_tensor = ChessEnv.encode_board(board)
    k = min(len(list(board.legal_moves)), 8)
    info_list = engine.analyse(board, chess.engine.Limit(depth=18), multipv=k)
    value_label = eval_to_value(info_list[0]["score"], board.turn)
    policy_label = sf_scores_to_probs_plane(info_list, board)
    action_mask = ChessEnv.create_plane_action_mask(board)
    action_mask = torch.tensor(action_mask, dtype=torch.float32)
    return (state_tensor, policy_label, value_label, action_mask), info_list

UNDERPROMO = {chess.ROOK, chess.BISHOP, chess.KNIGHT}
UNDERPROMO_SUFFIX = ("r", "b", "n")

def parse_chess_puzzles(engine, env, chunk_size=10_000, out_dir="Stockfish_test_data"):
    print("Reading Data Frame")
    df = pd.read_csv("lichess_db_puzzle.csv")

    os.makedirs(out_dir, exist_ok=True)

    memory = []
    files_saved = 0

    print("Starting Labeling")
    for row in df.itertuples(index=False):
        # adjust names if your columns differ
        fen = row.FEN
        moves_str = row.Moves

        move_sequence = moves_str.split()

        # 1) skip puzzles with underpromotion moves
        if any(uci.endswith(UNDERPROMO_SUFFIX) for uci in move_sequence):
            continue

        try:
            board = chess.Board(fen=fen)

            for uci in move_sequence:
                move = chess.Move.from_uci(uci)

                # extra safety: detect underpromotion via parsed move too
                if move.promotion in UNDERPROMO:
                    raise ValueError("Underpromotion detected")

                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move {uci} for FEN {fen}")

                board.push(move)

                memory.append(label_board(engine, board, env))

                if len(memory) >= chunk_size:
                    files_saved += 1
                    out_path = os.path.join(out_dir, f"labeled{files_saved}.pt")
                    torch.save(memory, out_path)
                    memory = []
                    print(f"Saved New File: {out_path}")

        except Exception:
            continue

    # save leftover
    if memory:
        files_saved += 1
        out_path = os.path.join(out_dir, f"labeled{files_saved}.pt")
        torch.save(memory, out_path)

ENGINE_PATH = r"C:\Users\Traedon Harris\Documents\GitHub\CS-4700-Final-Project\stockfish\stockfish-windows-x86-64-avx2.exe"

def open_and_init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        timeout=30.0,              # wait if DB is busy
        isolation_level=None,      # autocommit
        check_same_thread=False    # required for multiprocessing
    )

    # Concurrency + performance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    # Create table (safe to call every time)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        k INTEGER PRIMARY KEY,     -- board.transposition_key()
        value REAL NOT NULL,       -- value label
        moves TEXT NOT NULL,       -- JSON list of UCI moves
        policy_idx BLOB NOT NULL,  -- int32[]
        policy_prob BLOB NOT NULL  -- float32[]
    );
    """)

    return conn

def cache_get(conn: sqlite3.Connection, key: int):
    row = conn.execute(
        "SELECT value, moves, policy_idx, policy_prob FROM labels WHERE k=?",
        (key,)
    ).fetchone()
    if row is None:
        return None
    value, moves_json, idx_blob, prob_blob = row
    moves = json.loads(moves_json)
    idx_blob = np.frombuffer(idx_blob, dtype=np.int32).copy()
    prob_blob = np.frombuffer(prob_blob, dtype=np.float32).copy()
    idx, prob = unpack_sparse_policy(idx_blob, prob_blob)

    return float(value), moves, idx, prob

def cache_put(conn: sqlite3.Connection, key: int, value: float, moves: List[str],
              policy_idx: np.ndarray, policy_prob: np.ndarray) -> None:
    idx_blob, prob_blob = pack_sparse_policy(policy_idx, policy_prob)
    moves_json = json.dumps(moves)

    # INSERT OR IGNORE prevents races: if another worker wrote first, this is a no-op.
    conn.execute(
        "INSERT OR IGNORE INTO labels (k, value, moves, policy_idx, policy_prob) VALUES (?, ?, ?, ?, ?)",
        (key, float(value), moves_json, idx_blob, prob_blob)
    )

def pack_sparse_policy(action_indices: np.ndarray, action_probs: np.ndarray) -> Tuple[bytes, bytes]:
    # indices int32, probs float32
    action_indices = np.asarray(action_indices, dtype=np.int32)
    action_probs = np.asarray(action_probs, dtype=np.float32)
    return action_indices.tobytes(), action_probs.tobytes()

def unpack_sparse_policy(idx_blob: bytes, prob_blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.frombuffer(idx_blob, dtype=np.int32)
    prob = np.frombuffer(prob_blob, dtype=np.float32)
    return idx, prob

def sqlite_int64_from_uint64(x: int) -> int:
    if x >= (1 << 63):
        x -= (1 << 64)
    return x

def worker_generate(worker_id: int, n_files: int, length_of_file: int, out_dir: str, engine_path: str, db_path: str):
    os.makedirs(out_dir, exist_ok=True)

    conn = open_and_init_db(db_path)

    env = ChessEnv()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    memory = []
    files_saved = 0
    local_index = 0

    labeled_board_states = {}

    try:
        while files_saved < n_files:
            board = env.reset()
            while not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(board))
                hit = cache_get(conn, key)
                if hit is not None:
                    value_label, ranked_moves_uci, pol_idx, pol_prob = hit

                    state_tensor = ChessEnv.encode_board(board)
                    action_mask = torch.tensor(ChessEnv.create_plane_action_mask(board), dtype=torch.float32)

                    # rebuild dense policy plane tensor
                    policy_label = torch.zeros_like(action_mask)  # if same shape
                    policy_label.view(-1)[torch.from_numpy(pol_idx)] = torch.from_numpy(pol_prob)

                    label = (state_tensor, policy_label, value_label, action_mask)

                    ranked_moves = [chess.Move.from_uci(u) for u in ranked_moves_uci]

                else:
                    label, info_list = label_board(engine, board)
                    state_tensor, policy_label, value_label, action_mask = label

                    ranked_moves = set([info["pv"][0] for info in info_list if info.get("pv")])
                    ranked_moves_uci = [m.uci() for m in ranked_moves]

                    # compress policy to top-K (recommended)
                    flat = np.asarray(policy_label, dtype=np.float32).reshape(-1)
                    K = 32
                    top_idx = np.argpartition(-flat, K)[:K]
                    top_prob = flat[top_idx]
                    s = top_prob.sum()
                    if s > 0:
                        top_prob = top_prob / s

                    cache_put(conn, key, float(value_label), ranked_moves_uci, top_idx, top_prob)

                action = next(iter(ranked_moves))
                found_move = False
                for move in ranked_moves:
                    tmp_board = board.copy(stack=False)
                    tmp_board.push(move)
                    key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(tmp_board))
                    hit = cache_get(conn, key)
                    if hit is None:
                        action = move
                        found_move = True
                        break

                if not found_move:
                    for move in board.legal_moves:
                        if not move in ranked_moves:
                            tmp_board = board.copy(stack=False)
                            tmp_board.push(move)
                            key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(tmp_board))
                            hit = cache_get(conn, key)
                            if hit is None:
                                action = move
                                break

                memory.append(label)

                board.push(action)
                local_index += 1

            if len(memory) % 100 == 0:
                print(f"[Worker {worker_id}] {(len(memory) * 100) / length_of_file:.1f}% done")
            if len(memory) >= length_of_file:
                files_saved += 1
                out_path = os.path.join(out_dir, f"labeled_w{worker_id}_{files_saved}.pt")
                torch.save(memory, out_path)
                print(f"[Worker {worker_id}] Saved {out_path} ({len(memory)} samples)")
                memory = []

    finally:
        # flush leftovers
        if memory:
            files_saved += 1
            out_path = os.path.join(out_dir, f"labeled_w{worker_id}_{files_saved}.pt")
            torch.save(memory, out_path)
            print(f"[Worker {worker_id}] Saved leftover {out_path} ({len(memory)} samples)")
        engine.quit()

def parallel_data_gen(num_workers: int, files_per_worker: int, length_of_file: int, out_dir: str, engine_path: str, db_path: str):
    # On Windows: must protect the entry point
    ctx = mp.get_context("spawn")
    procs = []

    for wid in range(num_workers):
        p = ctx.Process(
            target=worker_generate,
            args=(wid, files_per_worker, length_of_file, out_dir, engine_path, db_path),
            daemon=False
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

def main():
    mp.freeze_support()  # safe on Windows
    parallel_data_gen(
        num_workers=10,          # start with #physical cores or slightly less
        files_per_worker=5,     # each worker writes this many files
        length_of_file=1000,    # samples per file (your length_of_file)
        out_dir="Stockfish_data_val",
        engine_path=ENGINE_PATH,
        db_path="stockfish_label_cache.db"
    )

if __name__ == "__main__":
    main()

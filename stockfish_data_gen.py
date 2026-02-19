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

from pathlib import Path

def get_engine_path() -> str:
    # folder where stockfish_data_gen.py lives
    root = Path(__file__).resolve().parent
    exe = root/"stockfish"/"stockfish-windows-x86-64-avx2.exe"
    return str(exe)

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
    k = min(len(list(board.legal_moves)), len(list(board.legal_moves)))
    info_list = engine.analyse(board, chess.engine.Limit(depth=5), multipv=k)
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
        timeout=30.0,
        isolation_level=None,
        check_same_thread=False
    )

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS labels (
        k INTEGER PRIMARY KEY,
        value REAL NOT NULL,
        moves TEXT NOT NULL,
        policy_idx BLOB NOT NULL,
        policy_prob BLOB NOT NULL,
        visits INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Migration for existing DB files
    try:
        conn.execute("ALTER TABLE labels ADD COLUMN visits INTEGER NOT NULL DEFAULT 0;")
    except sqlite3.OperationalError:
        # column already exists
        pass

    return conn

def cache_peek_visits(conn: sqlite3.Connection, key: int) -> int:
    row = conn.execute("SELECT visits FROM labels WHERE k=?", (key,)).fetchone()
    return int(row[0]) if row is not None else 0

def cache_increment_visits(conn: sqlite3.Connection, key: int) -> None:
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))


def cache_get(conn: sqlite3.Connection, key: int):
    row = conn.execute(
        "SELECT value, moves, policy_idx, policy_prob, visits FROM labels WHERE k=?",
        (key,)
    ).fetchone()
    if row is None:
        return None

    value, moves_json, idx_blob, prob_blob, visits = row

    # TOUCH: increment visits because we visited this state
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))

    moves = json.loads(moves_json)
    idx = np.frombuffer(idx_blob, dtype=np.int32).copy()
    prob = np.frombuffer(prob_blob, dtype=np.float32).copy()
    idx, prob = unpack_sparse_policy(idx, prob)

    return float(value), moves, idx, prob, int(visits) + 1

def cache_put(conn: sqlite3.Connection, key: int, value: float, moves: list[str],
              policy_idx: np.ndarray, policy_prob: np.ndarray) -> None:
    idx_blob, prob_blob = pack_sparse_policy(policy_idx, policy_prob)
    moves_json = json.dumps(moves)

    conn.execute(
        "INSERT OR IGNORE INTO labels (k, value, moves, policy_idx, policy_prob, visits) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (key, float(value), moves_json, idx_blob, prob_blob)
    )

    # TOUCH: count this as a visit
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))

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

def policy_prob_for_move(policy_label: torch.Tensor, board: chess.Board, move: chess.Move) -> float:
    plane, row, col = ChessEnv.encode_action(move, board)
    return float(policy_label[plane, row, col].item())

def select_move_least_visited_then_best(
    conn: sqlite3.Connection,
    board: chess.Board,
    candidate_moves: list[chess.Move],
    policy_label: torch.Tensor,
) -> chess.Move:
    best_move = None

    best_visits = None
    best_score = None  # higher is better for tie-break

    for mv in candidate_moves:
        tmp = board.copy(stack=False)
        tmp.push(mv)
        child_key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(tmp))

        v = cache_peek_visits(conn, child_key)  # NO touch
        s = policy_prob_for_move(policy_label, board, mv)  # Stockfish-derived probability

        if (best_move is None
            or v < best_visits
            or (v == best_visits and s > best_score)):
            best_move = mv
            best_visits = v
            best_score = s

    return best_move

def worker_generate(worker_id: int, n_files: int, length_of_file: int, out_dir: str, engine_path: str, db_path: str):
    os.makedirs(out_dir, exist_ok=True)

    conn = open_and_init_db(db_path)

    env = ChessEnv()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    memory = []
    files_saved = 0

    game_counter = 0

    try:
        while files_saved < n_files:
            board = env.reset()
            game_counter += 1
            move_count = 0
            print(f"[Worker {worker_id}] Starting Game: {game_counter}")
            while not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(board))
                hit = cache_get(conn, key)
                if hit is not None:
                    value_label, ranked_moves_uci, pol_idx, pol_prob, _ = hit

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

                    ranked_moves = list([info["pv"][0] for info in info_list if info.get("pv")])
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

                ranked_moves_list = list(ranked_moves) if ranked_moves else []
                candidates = ranked_moves_list + list(board.legal_moves)
                action = select_move_least_visited_then_best(conn, board, candidates, policy_label)

                memory.append(label)
                if move_count % 10 == 0:
                    print(f"[Worker {worker_id}] played: {action.uci()}, for the posistion: {board.fen()}")
                board.push(action)
                move_count += 1
                

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

def parallel_data_gen(num_workers: int, files_per_worker: int, length_of_file: int,
                      out_dir: str, engine_path: str, db_path: str,
                      stagger_seconds: float = 0.5):
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

        # Stagger starts
        if stagger_seconds > 0 and wid < num_workers - 1:
            time.sleep(stagger_seconds)

    for p in procs:
        p.join()

def main():
    mp.freeze_support()  # safe on Windows
    parallel_data_gen(
        num_workers=4,          # start with #physical cores or slightly less
        files_per_worker=4,     # each worker writes this many files
        length_of_file=1000,    # samples per file (your length_of_file)
        out_dir="Stockfish_data_val",
        engine_path=get_engine_path(),
        db_path="stockfish_label_cache.db",
        stagger_seconds=10
    )

if __name__ == "__main__":
    main()

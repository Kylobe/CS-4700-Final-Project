import chess
import torch
from ChessEnv import ChessEnv
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
from tqdm import tqdm
from torch.utils.data import Dataset

from pathlib import Path

DEFAULT_THINK_MS = 75
DEFAULT_PUZZLE_CHUNK_SIZE = 10000
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_ENGINE_INIT_TIMEOUT = 60.0
DEFAULT_ENGINE_INIT_RETRIES = 3
DEFAULT_ENGINE_INIT_RETRY_DELAY = 5.0

def get_engine_path() -> str:
    root = Path(__file__).resolve().parent
    exe = root/"stockfish"/"stockfish-windows-x86-64-avx2.exe"
    return str(exe)

def open_stockfish_engine(
    engine_path: str,
    timeout: float = DEFAULT_ENGINE_INIT_TIMEOUT,
    retries: int = DEFAULT_ENGINE_INIT_RETRIES,
    retry_delay: float = DEFAULT_ENGINE_INIT_RETRY_DELAY,
):
    last_error = None

    for attempt in range(retries):
        try:
            return chess.engine.SimpleEngine.popen_uci(engine_path, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(
        f"Failed to start Stockfish after {retries} attempts: {engine_path}"
    ) from last_error

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

def _get_analysis_game(engine):
    game = getattr(engine, "_analysis_game", None)
    if game is not None:
        return game

    game_cls = getattr(chess.engine, "Game", None)
    if game_cls is None:
        return None

    try:
        game = game_cls()
    except TypeError:
        try:
            game = game_cls
        except Exception:
            return None
    except Exception:
        return None

    try:
        setattr(engine, "_analysis_game", game)
    except Exception:
        pass

    return game

def label_board(engine, board: chess.Board, think_ms: int):
    state_tensor = ChessEnv.encode_board(board)
    k = min(len(list(board.legal_moves)), 8)
    limit = chess.engine.Limit(time=max(1, think_ms) / 1000.0)
    analyse_kwargs = {"multipv": k}

    game = _get_analysis_game(engine)
    if game is not None:
        analyse_kwargs["game"] = game

    try:
        info_list = engine.analyse(board, limit, **analyse_kwargs)
    except TypeError:
        analyse_kwargs.pop("game", None)
        info_list = engine.analyse(board, limit, **analyse_kwargs)

    if not isinstance(info_list, list):
        info_list = [info_list]

    score = info_list[0]["score"]
    value_label = eval_to_value(score, board.turn)
    policy_label = sf_scores_to_probs_plane(info_list, board)
    action_mask = ChessEnv.create_plane_action_mask(board)
    action_mask = torch.tensor(action_mask, dtype=torch.float32)
    return (state_tensor, policy_label, value_label, action_mask), info_list

def top_k_policy_entries(policy_label, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(policy_label, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    k = min(k, flat.size)
    top_idx = np.argpartition(-flat, k - 1)[:k].astype(np.int32, copy=False)
    top_prob = flat[top_idx].astype(np.float32, copy=False)
    total = float(top_prob.sum())
    if total > 0:
        top_prob = top_prob / total

    return top_idx.copy(), top_prob.copy()

def label_from_cache(board: chess.Board, value_label: float, pol_idx: np.ndarray, pol_prob: np.ndarray):
    state_tensor = ChessEnv.encode_board(board)
    action_mask = torch.tensor(ChessEnv.create_plane_action_mask(board), dtype=torch.float32)
    policy_label = torch.zeros_like(action_mask)
    if len(pol_idx) > 0:
        policy_label.view(-1)[torch.from_numpy(pol_idx)] = torch.from_numpy(pol_prob)
    return (state_tensor, policy_label, value_label, action_mask)

def should_assign_validation(board: chess.Board, validation_ratio: float = DEFAULT_VALIDATION_RATIO) -> bool:
    if validation_ratio <= 0:
        return False
    if validation_ratio >= 1:
        return True

    bucket_count = 10000
    threshold = int(validation_ratio * bucket_count)
    key = chess.polyglot.zobrist_hash(board)
    return (key % bucket_count) < threshold

def flush_memory_chunk(memory, out_dir: Path, prefix: str, files_saved: int) -> int:
    if not memory:
        return files_saved

    files_saved += 1
    out_path = out_dir / f"{prefix}{files_saved}.pt"
    torch.save(memory, out_path)
    memory.clear()
    return files_saved

def get_puzzle_csv_path() -> Path:
    return Path(__file__).resolve().parent / "lichess_db_puzzle.csv"

def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)

def worker_generate_puzzle_positions(
    worker_id: int,
    num_workers: int,
    total_rows: int,
    engine_path: str,
    db_path: str,
    think_ms: int,
    chunk_size: int,
    validation_ratio: float,
    train_dir: str,
    val_dir: str,
    save_shards: bool,
):
    csv_path = get_puzzle_csv_path()
    if num_workers <= 1:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(
            csv_path,
            skiprows=lambda i: i != 0 and ((i - 1) % num_workers != worker_id),
        )

    root = Path(__file__).resolve().parent
    train_path = root / train_dir
    val_path = root / val_dir
    if save_shards:
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

    conn = open_and_init_db(db_path)
    engine = open_stockfish_engine(engine_path)

    train_memory = []
    val_memory = []
    train_files_saved = 0
    val_files_saved = 0
    worker_total = (total_rows + num_workers - 1 - worker_id) // num_workers

    pbar = tqdm(
        total=max(worker_total, 0),
        desc=f"Puzzle Worker {worker_id}",
        position=worker_id,
        leave=False,
        dynamic_ncols=True
    )

    try:
        for row in df.itertuples(index=False):
            fen = row.FEN
            move_sequence = row.Moves.split()

            try:
                board = chess.Board(fen=fen)

                label, is_validation, _ = fetch_or_label_position(
                    conn=conn,
                    engine=engine,
                    board=board,
                    think_ms=think_ms,
                    validation_ratio=validation_ratio,
                )
                if is_validation:
                    if save_shards:
                        val_memory.append(label)
                else:
                    if save_shards:
                        train_memory.append(label)

                if save_shards and len(train_memory) >= chunk_size:
                    train_files_saved = flush_memory_chunk(train_memory, train_path, f"labeled_w{worker_id}_", train_files_saved)
                if save_shards and len(val_memory) >= chunk_size:
                    val_files_saved = flush_memory_chunk(val_memory, val_path, f"labeled_w{worker_id}_", val_files_saved)

                for uci in move_sequence:
                    move = chess.Move.from_uci(uci)

                    if move not in board.legal_moves:
                        raise ValueError(f"Illegal move {uci} for FEN {fen}")

                    board.push(move)

                    label, is_validation, _ = fetch_or_label_position(
                        conn=conn,
                        engine=engine,
                        board=board,
                        think_ms=think_ms,
                        validation_ratio=validation_ratio,
                    )
                    if is_validation:
                        if save_shards:
                            val_memory.append(label)
                    else:
                        if save_shards:
                            train_memory.append(label)

                    if save_shards and len(train_memory) >= chunk_size:
                        train_files_saved = flush_memory_chunk(train_memory, train_path, f"labeled_w{worker_id}_", train_files_saved)
                    if save_shards and len(val_memory) >= chunk_size:
                        val_files_saved = flush_memory_chunk(val_memory, val_path, f"labeled_w{worker_id}_", val_files_saved)

            except Exception:
                pass
            finally:
                pbar.update(1)

    finally:
        if save_shards:
            train_files_saved = flush_memory_chunk(train_memory, train_path, f"labeled_w{worker_id}_", train_files_saved)
            val_files_saved = flush_memory_chunk(val_memory, val_path, f"labeled_w{worker_id}_", val_files_saved)
        pbar.close()
        engine.quit()
        conn.close()

def worker_generate_puzzle_positions_entry(lock, *args):
    _init_tqdm(lock)
    worker_generate_puzzle_positions(*args)

def generate_puzzle_position_data(
    engine_path: str,
    db_path: str,
    think_ms: int = DEFAULT_THINK_MS,
    chunk_size: int = DEFAULT_PUZZLE_CHUNK_SIZE,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    train_dir: str = "Puzzle_Training_data",
    val_dir: str = "Puzzle_Validation_data",
    num_workers: int = 1,
    stagger_seconds: float = 2.0,
    save_shards: bool = False,
):
    csv_path = get_puzzle_csv_path()
    total_rows = count_csv_rows(csv_path)
    root = Path(__file__).resolve().parent
    train_path = root / train_dir
    val_path = root / val_dir
    if save_shards:
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

    if num_workers <= 1:
        worker_generate_puzzle_positions(
            worker_id=0,
            num_workers=1,
            total_rows=total_rows,
            engine_path=engine_path,
            db_path=db_path,
            think_ms=think_ms,
            chunk_size=chunk_size,
            validation_ratio=validation_ratio,
            train_dir=train_dir,
            val_dir=val_dir,
            save_shards=save_shards,
        )
    else:
        ctx = mp.get_context("spawn")
        lock = ctx.RLock()
        procs = []

        for worker_id in range(num_workers):
            p = ctx.Process(
                target=worker_generate_puzzle_positions_entry,
                args=(
                    lock,
                    worker_id,
                    num_workers,
                    total_rows,
                    engine_path,
                    db_path,
                    think_ms,
                    chunk_size,
                    validation_ratio,
                    train_dir,
                    val_dir,
                    save_shards,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

            if stagger_seconds > 0 and worker_id < num_workers - 1:
                time.sleep(stagger_seconds)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Puzzle worker exited with code {p.exitcode}")

    return {
        "csv_path": str(csv_path),
        "train_dir": str(train_path),
        "val_dir": str(val_path),
        "num_workers": num_workers,
        "save_shards": save_shards,
    }

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
        fen TEXT,
        value REAL NOT NULL,
        moves TEXT NOT NULL,
        policy_idx BLOB NOT NULL,
        policy_prob BLOB NOT NULL,
        visits INTEGER NOT NULL DEFAULT 0,
        is_validation_position INTEGER NOT NULL DEFAULT 0
    );
    """)

    # Migration for existing DB files
    try:
        conn.execute("ALTER TABLE labels ADD COLUMN visits INTEGER NOT NULL DEFAULT 0;")
    except sqlite3.OperationalError:
        # column already exists
        pass
    try:
        conn.execute("ALTER TABLE labels ADD COLUMN fen TEXT;")
    except sqlite3.OperationalError:
        # column already exists
        pass
    # Migration for is_validation_position column
    try:
        conn.execute("""
            ALTER TABLE labels
            ADD COLUMN is_validation_position INTEGER NOT NULL DEFAULT 0;
        """)
    except sqlite3.OperationalError:
        # column already exists
        pass

    return conn

def cache_peek_visits(conn: sqlite3.Connection, key: int) -> int:
    row = conn.execute("SELECT visits FROM labels WHERE k=?", (key,)).fetchone()
    return int(row[0]) if row is not None else 0

def cache_increment_visits(conn: sqlite3.Connection, key: int) -> None:
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))

def cache_mark_validation(conn: sqlite3.Connection, key: int):
    conn.execute(
        "UPDATE labels SET is_validation_position = 1 WHERE k=?",
        (key,)
    )

def cache_set_validation(conn: sqlite3.Connection, key: int, is_validation: bool):
    conn.execute(
        "UPDATE labels SET is_validation_position = ? WHERE k=?",
        (int(is_validation), key)
    )

def cache_get(conn: sqlite3.Connection, key: int):
    row = conn.execute(
        "SELECT fen, value, moves, policy_idx, policy_prob, visits, is_validation_position FROM labels WHERE k=?",
        (key,)
    ).fetchone()
    if row is None:
        return None

    fen, value, moves_json, idx_blob, prob_blob, visits, validation = row

    # TOUCH: increment visits because we visited this state
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))

    moves = json.loads(moves_json)
    idx = np.frombuffer(idx_blob, dtype=np.int32).copy()
    prob = np.frombuffer(prob_blob, dtype=np.float32).copy()
    idx, prob = unpack_sparse_policy(idx, prob)

    return fen, float(value), moves, idx, prob, int(visits) + 1, int(validation)

def cache_put(conn: sqlite3.Connection, key: int, value: float, moves: list[str],
              policy_idx: np.ndarray, policy_prob: np.ndarray, validation, fen: Optional[str] = None) -> None:
    idx_blob, prob_blob = pack_sparse_policy(policy_idx, policy_prob)
    moves_json = json.dumps(moves)

    cur = conn.execute(
        "INSERT OR IGNORE INTO labels (k, fen, value, moves, policy_idx, policy_prob, visits, is_validation_position) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        (key, fen, float(value), moves_json, idx_blob, prob_blob, validation)
    )

    inserted = (cur.rowcount == 1)

    if fen is not None:
        conn.execute("UPDATE labels SET fen = COALESCE(fen, ?) WHERE k=?", (fen, key))

    # TOUCH: count this as a visit
    conn.execute("UPDATE labels SET visits = visits + 1 WHERE k=?", (key,))
    return inserted

def pack_sparse_policy(action_indices: np.ndarray, action_probs: np.ndarray) -> Tuple[bytes, bytes]:
    # indices int32, probs float32
    action_indices = np.asarray(action_indices, dtype=np.int32)
    action_probs = np.asarray(action_probs, dtype=np.float32)
    return action_indices.tobytes(), action_probs.tobytes()

def unpack_sparse_policy(idx_blob: bytes, prob_blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.frombuffer(idx_blob, dtype=np.int32)
    prob = np.frombuffer(prob_blob, dtype=np.float32)
    return idx, prob

def row_to_training_sample(fen: str, value: float, pol_idx: np.ndarray, pol_prob: np.ndarray):
    board = chess.Board(fen=fen)
    state_tensor = ChessEnv.encode_board(board)
    action_mask = torch.tensor(ChessEnv.create_plane_action_mask(board), dtype=torch.float32)
    policy_label = torch.zeros_like(action_mask)
    if len(pol_idx) > 0:
        policy_label.view(-1)[torch.from_numpy(pol_idx)] = torch.from_numpy(pol_prob)
    return state_tensor, policy_label, float(value), action_mask

def count_db_samples(db_path: str, validation: Optional[bool] = None) -> int:
    conn = open_and_init_db(db_path)
    try:
        if validation is None:
            row = conn.execute("SELECT COUNT(*) FROM labels WHERE fen IS NOT NULL").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM labels WHERE is_validation_position = ? AND fen IS NOT NULL",
                (int(validation),)
            ).fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        conn.close()

class StockfishLabelDBDataset(Dataset):
    def __init__(self, db_path: str, validation: bool = False):
        self.db_path = db_path
        self.validation = validation
        self.conn = open_and_init_db(db_path)
        self.keys = [
            row[0] for row in self.conn.execute(
                "SELECT k FROM labels WHERE is_validation_position = ? AND fen IS NOT NULL ORDER BY k",
                (int(validation),)
            ).fetchall()
        ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        row = self.conn.execute(
            "SELECT fen, value, policy_idx, policy_prob FROM labels WHERE k=?",
            (key,)
        ).fetchone()
        if row is None:
            raise IndexError(f"Missing label row for key {key}")

        fen, value, idx_blob, prob_blob = row
        pol_idx, pol_prob = unpack_sparse_policy(idx_blob, prob_blob)
        return row_to_training_sample(fen, value, pol_idx.copy(), pol_prob.copy())

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()

def sqlite_int64_from_uint64(x: int) -> int:
    if x >= (1 << 63):
        x -= (1 << 64)
    return x

def fetch_or_label_position(
    conn: sqlite3.Connection,
    engine,
    board: chess.Board,
    think_ms: int,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
):
    key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(board))
    desired_validation = should_assign_validation(board, validation_ratio)
    hit = cache_get(conn, key)

    if hit is not None:
        fen, value_label, ranked_moves_uci, pol_idx, pol_prob, _, is_validation = hit
        if bool(is_validation) != desired_validation:
            cache_set_validation(conn, key, desired_validation)
            is_validation = int(desired_validation)
        if fen is None:
            conn.execute("UPDATE labels SET fen = ? WHERE k=?", (board.fen(), key))
        label = label_from_cache(board, value_label, pol_idx, pol_prob)
        return label, bool(is_validation), ranked_moves_uci

    label, info_list = label_board(engine, board, think_ms)
    _, policy_label, value_label, _ = label

    ranked_moves = [info["pv"][0] for info in info_list if info.get("pv")]
    ranked_moves_uci = [mv.uci() for mv in ranked_moves]
    top_idx, top_prob = top_k_policy_entries(policy_label)
    is_validation = desired_validation

    cache_put(
        conn,
        key,
        float(value_label),
        ranked_moves_uci,
        top_idx,
        top_prob,
        validation=int(is_validation),
        fen=board.fen(),
    )

    return label, is_validation, ranked_moves_uci

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

def worker_generate(worker_id: int, n_files: int, length_of_file: int, out_dir: str, engine_path: str, db_path: str, creating_validation: bool = False, think_ms: int = DEFAULT_THINK_MS):
    os.makedirs(out_dir, exist_ok=True)

    conn = open_and_init_db(db_path)

    env = ChessEnv()
    engine = open_stockfish_engine(engine_path)

    memory = []
    files_saved = 0

    pbar = tqdm(
        total=n_files,
        desc=f"Worker {worker_id}",
        position=worker_id,
        leave=False,
        dynamic_ncols=True
    )

    game_counter = 0

    try:
        while files_saved < n_files:
            board = env.reset()
            game_counter += 1
            move_count = 0
            #if (game_counter - 1) % 10 == 0:
                #print(f"[Worker {worker_id}] Starting Game: {game_counter}")
            while not board.is_game_over():
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                key = sqlite_int64_from_uint64(chess.polyglot.zobrist_hash(board))
                hit = cache_get(conn, key)

                if hit is not None:
                    _, value_label, ranked_moves_uci, pol_idx, pol_prob, _, is_validation = hit

                    state_tensor = ChessEnv.encode_board(board)
                    action_mask = torch.tensor(ChessEnv.create_plane_action_mask(board), dtype=torch.float32)

                    policy_label = torch.zeros_like(action_mask)
                    policy_label.view(-1)[torch.from_numpy(pol_idx)] = torch.from_numpy(pol_prob)

                    label = (state_tensor, policy_label, value_label, action_mask)
                    ranked_moves = [chess.Move.from_uci(u) for u in ranked_moves_uci]

                    # Add to memory only if we're NOT creating validation and this pos is NOT validation
                    if (not creating_validation) and (not is_validation):
                        memory.append(label)

                else:
                    # Miss: label with Stockfish
                    label, info_list = label_board(engine, board, think_ms)
                    state_tensor, policy_label, value_label, action_mask = label

                    ranked_moves = [info["pv"][0] for info in info_list if info.get("pv")]
                    ranked_moves_uci = [m.uci() for m in ranked_moves]

                    flat = np.asarray(policy_label, dtype=np.float32).reshape(-1)
                    K = 32
                    top_idx = np.argpartition(-flat, K)[:K]
                    top_prob = flat[top_idx]
                    s = top_prob.sum()
                    if s > 0:
                        top_prob = top_prob / s

                    # Insert and detect if it truly was "unique"
                    inserted = cache_put(
                        conn, key, float(value_label), ranked_moves_uci, top_idx, top_prob,
                        validation=int(creating_validation)
                    )

                    # If creating validation: only keep it if we actually inserted it (unique)
                    if creating_validation:
                        if inserted:
                            memory.append(label)
                        # else: someone else inserted it first -> NOT unique -> skip
                    else:
                        # training mode: new misses are not validation, so keep
                        memory.append(label)
                        

                ranked_moves_list = list(ranked_moves) if ranked_moves else []
                candidates = ranked_moves_list + list(board.legal_moves)
                action = select_move_least_visited_then_best(conn, board, candidates, policy_label)
                board.push(action)
                move_count += 1
                
            #if (game_counter - 1) % 10 == 0:
                #print(f"[Worker {worker_id}] {((len(memory) + files_saved * length_of_file) * 100) / (length_of_file * n_files):.1f}% done")
            if len(memory) >= length_of_file:
                files_saved += 1
                out_path = os.path.join(out_dir, f"labeled_w{worker_id}_{files_saved}.pt")
                torch.save(memory, out_path)
                #print(f"[Worker {worker_id}] Saved {out_path} ({len(memory)} samples) | {((len(memory) + files_saved * length_of_file) * 100) / (length_of_file * n_files):.1f}% done")
                memory = []
                pbar.update(1)

    finally:
        if memory:
            files_saved += 1
            out_path = os.path.join(out_dir, f"labeled_w{worker_id}_{files_saved}.pt")
            torch.save(memory, out_path)
            #print(f"[Worker {worker_id}] Saved leftover {out_path} ({len(memory)} samples)")
        pbar.close()
        engine.quit()

def worker_entry(lock, *args):
    _init_tqdm(lock)
    worker_generate(*args)

def parallel_data_gen(num_workers: int, files_per_worker: int, length_of_file: int,
                      out_dir: str, engine_path: str, db_path: str,
                      stagger_seconds: float = 0.5, creating_validation: bool = False,
                      think_ms: int = DEFAULT_THINK_MS):
    ctx = mp.get_context("spawn")
    lock = ctx.RLock()
    procs = []

    for wid in range(num_workers):
        p = ctx.Process(
            target=worker_entry,
            args=(lock, wid, files_per_worker, length_of_file, out_dir, engine_path, db_path, creating_validation, think_ms),
            daemon=False
        )
        p.start()
        procs.append(p)

        # Stagger starts
        if stagger_seconds > 0 and wid < num_workers - 1:
            time.sleep(stagger_seconds)

    for p in procs:
        p.join()

_TQDM_LOCK = None

def _init_tqdm(lock):
    global _TQDM_LOCK
    _TQDM_LOCK = lock
    tqdm.set_lock(lock)

def main():
    """
    mp.freeze_support()  # safe on Windows
    parallel_data_gen(
        num_workers=10,          # start with #physical cores or slightly less
        files_per_worker=20,     # each worker writes this many files
        length_of_file=1000,    # samples per file (your length_of_file)
        out_dir="Stockfish_test_data",
        engine_path=get_engine_path(),
        db_path="stockfish_label_cache.db",
        stagger_seconds=2,
        think_ms=DEFAULT_THINK_MS,
    )
    """
    num_workers = max(1, min(12, mp.cpu_count()))
    print(f"Starting Puzzle Labeling With {num_workers} Workers")

    generate_puzzle_position_data(
        get_engine_path(),
        "stockfish_label_cache.db",
        num_workers=num_workers,
    )

if __name__ == "__main__":
    main()

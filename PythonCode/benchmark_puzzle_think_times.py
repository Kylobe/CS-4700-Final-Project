import csv
import random
import time
from pathlib import Path
from typing import Iterable

import chess
import chess.engine

from stockfish_data_gen import (
    UNDERPROMO,
    UNDERPROMO_SUFFIX,
    get_engine_path,
    get_puzzle_csv_path,
    open_stockfish_engine,
)


DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 4700
DEFAULT_THINK_TIMES_MS = [25, 50, 75, 100, 250, 500, 1000, 2000]


def reservoir_sample_puzzles(
    csv_path: Path,
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        seen = 0

        for row in reader:
            moves_str = row.get("Moves", "")
            if not moves_str:
                continue

            move_sequence = moves_str.split()
            if any(uci.endswith(UNDERPROMO_SUFFIX) for uci in move_sequence):
                continue

            seen += 1
            if len(sample) < sample_size:
                sample.append(row)
                continue

            replace_idx = rng.randrange(seen)
            if replace_idx < sample_size:
                sample[replace_idx] = row

    return sample


def analyze_best_move(engine, board: chess.Board, think_ms: int) -> chess.Move:
    limit = chess.engine.Limit(time=max(1, think_ms) / 1000.0)
    k = min(len(list(board.legal_moves)), 8)
    info_list = engine.analyse(board, limit, multipv=k)
    pv = info_list[0].get("pv")
    if not pv:
        return None
    return pv[0]


def evaluate_puzzle(engine, fen: str, moves_str: str, think_ms: int) -> dict[str, object]:
    move_sequence = moves_str.split()
    if len(move_sequence) < 2:
        return None

    try:
        board = chess.Board(fen=fen)
    except ValueError:
        return None

    opening_move = chess.Move.from_uci(move_sequence[0])
    if opening_move not in board.legal_moves:
        return None

    # The initial FEN is the position before the opponent's blunder.
    # Apply that move first so we benchmark the solver's response.
    board.push(opening_move)

    solver_side = board.turn
    solver_turns = 0
    matched_solver_turns = 0
    first_move_correct = False
    line_solved = True

    started_at = time.perf_counter()

    for solver_ply_index, expected_uci in enumerate(move_sequence[1:]):
        expected_move = chess.Move.from_uci(expected_uci)

        if expected_move not in board.legal_moves:
            return None

        if board.turn == solver_side:
            solver_turns += 1
            predicted_move = analyze_best_move(engine, board, think_ms)
            if predicted_move is None:
                line_solved = False
                break

            is_match = predicted_move == expected_move
            if solver_ply_index == 0:
                first_move_correct = is_match
            if is_match:
                matched_solver_turns += 1
            else:
                line_solved = False
                break

        board.push(expected_move)

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    return {
        "first_move_correct": first_move_correct,
        "line_solved": line_solved,
        "solver_turns": solver_turns,
        "matched_solver_turns": matched_solver_turns,
        "elapsed_ms": elapsed_ms,
    }


def benchmark_think_times(
    think_times_ms: Iterable[int],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
):
    csv_path = get_puzzle_csv_path()
    sample = reservoir_sample_puzzles(csv_path, sample_size=sample_size, seed=seed)

    if not sample:
        raise RuntimeError(f"No puzzles sampled from {csv_path}")

    print(f"Sampled {len(sample)} puzzles from {csv_path.name} with seed={seed}")

    engine = open_stockfish_engine(get_engine_path())

    try:
        for think_ms in think_times_ms:
            tested = 0
            first_move_hits = 0
            line_solved_hits = 0
            total_solver_turns = 0
            matched_solver_turns = 0
            total_elapsed_ms = 0.0

            for row in sample:
                result = evaluate_puzzle(
                    engine=engine,
                    fen=row["FEN"],
                    moves_str=row["Moves"],
                    think_ms=think_ms,
                )
                if result is None:
                    continue

                tested += 1
                first_move_hits += int(result["first_move_correct"])
                line_solved_hits += int(result["line_solved"])
                total_solver_turns += int(result["solver_turns"])
                matched_solver_turns += int(result["matched_solver_turns"])
                total_elapsed_ms += float(result["elapsed_ms"])

            if tested == 0:
                print(f"think_ms={think_ms}: no valid puzzles tested")
                continue

            first_move_acc = first_move_hits / tested
            line_solve_acc = line_solved_hits / tested
            per_turn_acc = matched_solver_turns / max(total_solver_turns, 1)
            avg_ms_per_puzzle = total_elapsed_ms / tested

            print(
                f"think_ms={think_ms:>4} | "
                f"tested={tested:>3} | "
                f"first_move_acc={first_move_acc:.3f} | "
                f"line_solve_acc={line_solve_acc:.3f} | "
                f"solver_turn_acc={per_turn_acc:.3f} | "
                f"avg_ms_per_puzzle={avg_ms_per_puzzle:.1f}"
            )

            if per_turn_acc == 1 and first_move_acc == 1 and line_solve_acc == 1:
                print(f"think_ms= {think_ms} had a perfect run")
                break
    finally:
        engine.quit()


def main():
    benchmark_think_times(DEFAULT_THINK_TIMES_MS)


if __name__ == "__main__":
    main()

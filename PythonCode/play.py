import torch
import chess
import chess.engine
import random
from ChessEnv import ChessEnv
import numpy as np
from AlphaZeroChess import AlphaZeroChess
from MCTS import MCTS

class ChessBot:
    def __init__(self, env:ChessEnv):
        self.env = env

    def get_best_move(self, board: chess.Board):
        pass

class MCTSBot(ChessBot):
    def __init__(self, env, args, model):
        super().__init__(env)
        self.mcts = MCTS(env=env, args=args, model=model)

    def get_best_move(self, board):
        action_probs = self.mcts.search(board, board.turn == chess.WHITE)
        action = np.random.choice(self.env.num_moves, p=action_probs)
        return action

class RandomBot(ChessBot):
    def __init__(self, env):
        super().__init__(env)

    def get_best_move(self, board):
        move = random.choice(list(board.legal_moves))
        move.promotion = None
        return self.env.encode_action(move, board.turn)

def random_vs_mcts(n, args):
    chess_env = ChessEnv()

    # White is random
    white_bot = RandomBot(chess_env)

    # Load model for black (MCTS)
    model = AlphaZeroChess(chess_env, num_resBlocks=args['res_blocks'], num_hidden=128)
    model.load_state_dict(torch.load("LatestAlphaChess.pt", map_location=torch.device('cpu')))
    model.eval()

    black_bot = MCTSBot(chess_env, args, model)

    # Counters
    white_wins = 0
    black_wins = 0
    draws = 0

    for game_idx in range(n):
        done = False
        chess_env.reset()

        # Reset MCTS tree for each game
        black_bot.mcts.root = None

        while not done:
            board = chess_env.board

            if board.turn == chess.WHITE:
                action_idx = white_bot.get_best_move(board)
            else:
                action_idx = black_bot.get_best_move(board)

            _, done = chess_env.step(action_idx)
            if black_bot.mcts.root is not None:
                black_bot.mcts.advance_root(action_idx)

        # Game finished
        result = chess_env.board.result()

        if result == "1-0":
            white_wins += 1
        elif result == "0-1":
            black_wins += 1
        else:
            draws += 1
            result = "1/2-1/2"

        print(f"Game {game_idx+1}/{n} finished: {result}")

    # Final summary
    print("\n================ RESULTS ================")
    print(f"Total games: {n}")
    print(f"White (Random) wins: {white_wins}")
    print(f"Black (MCTS) wins : {black_wins}")
    print(f"Draws            : {draws}")
    print("==========================================\n")

    return white_wins, black_wins, draws


def main():
    args = {
        'C': 2,
        'material_weight': 0.5,
        'num_searches': 200,
        'num_iterations': 10,
        'num_self_play_iterations': 5,
        'epochs': 5,
        'num_processes': 6,
        'res_blocks': 12,
        'num_hidden': 128,
        'batch_size': 128,
        'phase': 0,
        'syzygy_path':"C:\chess_tb\syzygy",
        'base-lr': 0.1,
        'use_tables':False
    }
    random_vs_mcts(10, args)

if __name__ == "__main__":
    main()


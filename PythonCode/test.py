from MCTS import MCTS
from ChessEnv import ChessEnv
from AlphaZeroChess import AlphaZeroChess, AlphaZero
import chess
import torch
from torch import optim
import numpy as np
from train_value import eval_to_value
import chess.engine


def test_engine(fen, engine):
    board = chess.Board(fen=fen)
    
    info = engine.analyse(board, chess.engine.Limit(depth=5))
    value_label = eval_to_value(info["score"], board.turn)
    print(value_label)

def test_value_head(fen, model):
    board = chess.Board(fen=fen)
    state_input = ChessEnv._encode_board(board).unsqueeze(0).to(model.device)
    policy, value_t = model(state_input.to(model.device))
    print(policy.shape)
    print(value_t.item())


def test_mcts(fen):
    args = {
        'C': 2,
        'material_weight': 0.5,
        'num_searches': 8000,
        'num_iterations': 10,
        'num_self_play_iterations': 5,
        'epochs': 5,
        'num_processes': 1,
        'res_blocks': 40,
        'num_hidden': 256,
        'batch_size': 256,
        'phase': 0,
        'syzygy_path':"C:\chess_tb\syzygy",
        'base-lr': 0.1,
        'use_tables':False,
    }
    env = ChessEnv()
    env.board = chess.Board(fen=fen)
    model = AlphaZeroChess(env, num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    model.load_state_dict(torch.load("PretrainModel.pt", map_location=torch.device('cpu')))
    #model.share_memory()
    mcts = MCTS(env=env, args=args, model=model)
    action_probs = mcts.search(env.board)
    action = np.unravel_index(np.argmax(action_probs), (73, 8, 8))
    mcts.advance_root(action)
    uci_action = ChessEnv.decode_action(action, env.board)
    print(uci_action)


def main():
    test_mcts("r1bqkb1r/ppp2ppp/2np1n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    #board = chess.Board(fen="k1r5/pp6/8/8/8/8/8/2R4K w - - 4 4")
    #print(ChessEnv.encode_action(chess.Move.from_uci("c1c8"), board))



if __name__ == "__main__":
    main()





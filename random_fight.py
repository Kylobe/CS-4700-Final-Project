from ChessEnv import ChessEnv
from AlphaZeroChess import AlphaZeroChess
from MCTS import MCTS
import torch
import numpy as np
import random
import chess

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_self_play_iterations': 10,
    'epochs': 4,
    'num_processes': 6,
    'res_blocks': 12,
    'num_hidden': 128,
    'batch_size': 64
}


env = ChessEnv()
state = env.reset()
done = False

# Your trained model
model = AlphaZeroChess(env=env, num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])

model.load_state_dict(torch.load("AlphaChess2.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()
mcts = MCTS(env=env, args=args, model=model)


def model_vs_random():
    player = True
    done = False
    model_is_white = True
    state = env.reset()
    cur_move = 1
    while not done:
        if player == model_is_white:
            action_probs = mcts.search(state, player)
            action = np.argmax(action_probs)
        else:
            legal_moves = list(env.board.legal_moves)
            move = random.choice(legal_moves)
            action = env.encode_action(move, env.board.turn)
        state, done = env.step(action)
        player = not player
        cur_move += 1
        if cur_move % 50 == 0:
            env.render()

    return env.board.result()


for _ in range(5):
    result = model_vs_random()
    print(result)

env.close()

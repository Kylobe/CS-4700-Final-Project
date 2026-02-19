from ChessEnv import ChessEnv
from AlphaZeroChess import AlphaZeroChess
from MCTS import MCTS
import torch
import numpy as np
import random
import chess

args = {
    'C': 2,
    'num_searches': 200,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'res_blocks': 40,
    'num_hidden': 256,
}

env = ChessEnv()
state = env.reset()
done = False

# Your trained model
model = AlphaZeroChess(env=env, num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])

#model.load_state_dict(torch.load("PretrainModel.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()
mcts = MCTS(env=env, args=args, model=model)


def model_vs_random():
    player = True
    done = False
    model_is_white = True
    state = env.reset()
    mcts.root = None
    cur_move = 1
    while not done:
        print(f"Starting Turn: {cur_move}")
        if player == model_is_white:
            action_probs = mcts.search(state)
            flat_index = np.argmax(action_probs)
            action = np.unravel_index(flat_index, action_probs.shape)
        else:
            legal_moves = list(env.board.legal_moves)
            move = random.choice(legal_moves)
            action = ChessEnv.encode_action(move, env.board)
        state, done = env.step(action)
        mcts.advance_root(action)
        player = not player
        cur_move += 1
        if cur_move % 50 == 0:
            env.render()

    return env.board.result()


for _ in range(5):
    result = model_vs_random()
    print(result)

env.close()

import requests
import chess

def get_best_move(board:chess.Board):
    fen = board.fen().replace(" ", "_")
    r = requests.get(f"http://tablebase.lichess.ovh/standard?fen={fen}", timeout=3)
    if r.status_code == 200:
        data = r.json()
        cat = data["category"]          # 'win'/'draw'/'loss'/...
        best = data["moves"][0]["uci"]  # TB-best move
        print(best)
    else:
        return


import torch
import chess
from ChessNet import AlphaZeroChess
from ChessEnv import ChessEnv

CONFIG = {
    "res_blocks": 40,
    "num_hidden": 256,
    "model_path": "FiftyMillionPos.pt",
    "top_k": 5,
}


def load_model():
    model = AlphaZeroChess(
        num_resBlocks=CONFIG["res_blocks"],
        num_hidden=CONFIG["num_hidden"]
    )

    state_dict = torch.load(CONFIG["model_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def board_to_tensor(board):
    x = ChessEnv.encode_board(board)

    if isinstance(x, torch.Tensor):
        x = x.unsqueeze(0).float()
    else:
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    return x


def interpret_value(value_scalar, board):
    """
    Assumes value is in [-1, 1] from the perspective of side to move.
    Positive = side to move is better.
    Negative = side to move is worse.
    """
    score = float(value_scalar)

    if score > 0.2:
        side_text = "White" if board.turn == chess.WHITE else "Black"
        status = f"{side_text} is winning"
    elif score < -0.2:
        side_text = "Black" if board.turn == chess.WHITE else "White"
        status = f"{side_text} is winning"
    else:
        status = "Position looks roughly equal"

    return score, status


def get_move_probability(policy_tensor, board):
    """
    Returns a list of (move, probability) for all legal moves.

    Assumes:
      - policy_tensor shape is (1, 73, 8, 8) or (73, 8, 8)
      - ChessEnv.encode_action(move, board) returns (plane, row, col)
    """

    if policy_tensor.dim() == 4:
        policy_tensor = policy_tensor[0]   # now shape: (73, 8, 8)

    # Flatten, softmax over all action logits, then reshape back
    policy = torch.softmax(policy_tensor.reshape(-1), dim=0).reshape(policy_tensor.shape)

    # Convert legal move mask to tensor
    mask = ChessEnv.create_plane_action_mask(board)
    mask = torch.tensor(mask, dtype=policy.dtype, device=policy.device)

    # Zero out illegal moves
    policy = policy * mask

    # Renormalize over legal moves only
    s = policy.sum()
    if s > 0:
        policy = policy / s
    else:
        policy = mask / max(mask.sum().item(), 1)

    legal_move_probs = []

    for move in board.legal_moves:
        action = ChessEnv.encode_action(move, board)

        if isinstance(action, tuple) and len(action) == 3:
            plane, row, col = action
            prob = float(policy[plane, row, col].item())
            legal_move_probs.append((move, prob))
        else:
            raise ValueError(
                f"ChessEnv.encode_action(move, board) returned {action}, "
                "but expected (plane, row, col)."
            )

    legal_move_probs.sort(key=lambda x: x[1], reverse=True)
    return legal_move_probs


def print_top_moves(move_probs, top_k=5):
    print(f"\nTop {min(top_k, len(move_probs))} moves:")
    for i, (move, prob) in enumerate(move_probs[:top_k], start=1):
        print(f"{i}. {move.uci()}   probability = {prob:.6f}")


def main():
    model = load_model()

    # Starting position. Replace with a FEN if you want:
    board = chess.Board()

    # Example custom FEN:
    board = chess.Board("r1bqkb1r/ppp2ppp/2np1n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 1")

    print("Board:")
    print(board)

    x = board_to_tensor(board)

    with torch.no_grad():
        policy, value = model(x)

    value_scalar = value.squeeze().item()
    score, status = interpret_value(value_scalar, board)

    print("\nEvaluation:")
    print(f"Raw value: {score:.4f}")
    print(status)

    move_probs = get_move_probability(policy, board)
    print_top_moves(move_probs, CONFIG["top_k"])


if __name__ == "__main__":
    main()
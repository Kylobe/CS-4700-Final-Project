import chess
import chess.pgn
import random

def generate_endgame_board(pieces: dict, randomize_positions=True) -> chess.Board:
    """
    Generate a chess board with a specific set of endgame pieces.
    
    Args:
        pieces (dict): Dictionary with format {"wK": 1, "bK": 1, "wQ": 1, ...}
        randomize_positions (bool): If True, pieces are placed randomly on legal squares.
        
    Returns:
        chess.Board: A valid chess.Board object.
    """
    board = chess.Board()
    board.clear()  # Clear the board

    all_squares = list(chess.SQUARES)
    random.shuffle(all_squares)

    piece_map = {
        'K': chess.KING,
        'Q': chess.QUEEN,
        'R': chess.ROOK,
        'B': chess.BISHOP,
        'N': chess.KNIGHT,
        'P': chess.PAWN,
    }

    used_squares = set()

    for key, count in pieces.items():
        color = chess.WHITE if key[0] == 'w' else chess.BLACK
        piece_type = piece_map[key[1].upper()]
        for _ in range(count):
            for square in all_squares:
                if square in used_squares:
                    continue
                # Avoid illegal pawn placements
                if piece_type == chess.PAWN and (chess.square_rank(square) in [0, 7]):
                    continue
                board.set_piece_at(square, chess.Piece(piece_type, color))
                used_squares.add(square)
                break

    # Ensure kings are not adjacent or in check
    if not board.is_valid() or board.is_game_over():
        return generate_endgame_board(pieces, randomize_positions)
    
    board.turn = chess.WHITE
    return board

def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    """
    Check if a pawn at `square` is a passed pawn.
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    step = 1 if color == chess.WHITE else -1

    for f in [file - 1, file, file + 1]:
        if 0 <= f < 8:
            for r in range(rank + step, 8 if color == chess.WHITE else -1, step):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
    return True

def generate_passed_pawn_endgame(max_attempts=100) -> chess.Board:
    """
    Generate an endgame board with a passed pawn and kings.
    Tries up to `max_attempts` to find a valid position.
    """
    for _ in range(max_attempts):
        board = chess.Board()
        board.clear()
        
        # Place kings randomly
        king_squares = random.sample(list(chess.SQUARES), 2)
        board.set_piece_at(king_squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(king_squares[1], chess.Piece(chess.KING, chess.BLACK))

        used_squares = set(king_squares)

        # Try to place a white passed pawn
        candidate_squares = [sq for sq in chess.SQUARES if chess.square_rank(sq) not in (0, 7)]
        random.shuffle(candidate_squares)

        for sq in candidate_squares:
            if sq in used_squares:
                continue
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
            if is_passed_pawn(board, sq, chess.WHITE):
                board.turn = chess.WHITE
                return board
            else:
                board.remove_piece_at(sq)
    raise ValueError("Failed to generate a passed pawn endgame.")

# Example usage
if __name__ == "__main__":
    queen_endgame = {
        "wK": 1,
        "bK": 1,
        "wQ": 1,
    }

    rook_endgame = {
        "wK": 1,
        "bK": 1,
        "wR": 1,
    }

    bishop_knight_endgame = {
        "wK": 1,
        "bK": 1,
        "wB": 1,
        "wN": 1,
    }
    
    BB_endgame = {
        "wB1": 1,
        "wB2": 1,
        "wK": 1,
        "bK": 1,
    }

    QR_endgame = {
        "wQ": 1,
        "wR": 1,
        "wK": 1,
        "bK": 1,
    }

    NP_endgame = {
        "wN": 1,
        "wP": 1,
        "wK": 1,
        "bK": 1,
    }



    board = generate_endgame_board(queen_endgame)
    endgame_dicts = [queen_endgame, rook_endgame, bishop_knight_endgame, BB_endgame, QR_endgame, NP_endgame]
    endgame_fens = []
    for _ in range(1000):
        board = generate_endgame_board(random.choice(endgame_dicts))
        endgame_fens.append(f"{board.fen()}\n")

    with open("endgame_fens.txt", 'w', encoding='utf-8') as file:
        file.writelines(endgame_fens)

import chess

# --- Helpers ---
def square_to_rc(sq: int):
    # python-chess: file 0..7, rank 0..7
    return chess.square_rank(sq), chess.square_file(sq)

def uci_to_move(uci: str):
    return chess.Move.from_uci(uci)

# Directions for sliding planes: (dr, dc) with row=ranks
SLIDE_DIRS = [
    (+1,  0),  # N
    (-1,  0),  # S
    ( 0, +1),  # E
    ( 0, -1),  # W
    (+1, +1),  # NE
    (+1, -1),  # NW
    (-1, +1),  # SE
    (-1, -1),  # SW
]

KNIGHT_DELTAS = [
    (+2, +1), (+2, -1),
    (-2, +1), (-2, -1),
    (+1, +2), (+1, -2),
    (-1, +2), (-1, -2),
]

# Pawn underpromotion planes: base index 64
# For a mover, "forward" is +1 row for White, -1 row for Black.
# We'll encode them in a fixed order:
# promo piece in (N,B,R) and dir in (forward, cap_left, cap_right)
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
PROMO_DIRS = ["F", "L", "R"]  # forward, capture-left, capture-right

def encode_move_plane_row_col(turn: bool, move: chess.Move):
    """
    Returns (plane, row, col) for the given move on the given board.
    row,col correspond to from-square.
    """
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
        move = chess.Move(from_sq, to_sq, promotion=move.promotion)
    fr = move.from_square
    to = move.to_square
    r0, c0 = square_to_rc(fr)
    r1, c1 = square_to_rc(to)
    dr = r1 - r0
    dc = c1 - c0

    # Normalize "forward" direction based on side to move
    # White forward: +1 row, Black forward: -1 row
    fwd = 1

    # 1) Underpromotions (to N/B/R) use special planes 64-72
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        # Determine direction type relative to mover
        # Forward: (dr == fwd, dc == 0)
        # Capture-left/right: (dr == fwd, dc == -1/+1) for White
        # For Black, left/right swap in terms of dc sign, but we define L as "capture to file-1 from White POV".
        # Easiest: define L/R relative to the mover's perspective:
        # - "left" means dc == -1 for White, dc == +1 for Black (because you're facing the other way).
        if dr != fwd:
            raise ValueError("Invalid promotion delta (must advance 1)")

        if dc == 0:
            dir_idx = 0  # F
        else:
            # mover-relative left/right
            if dc == -1:
                dir_idx = 1  # L
            elif dc == 1:
                dir_idx = 2  # R
            else:
                raise ValueError("Invalid promotion capture delta")

        promo_idx = PROMO_PIECES.index(move.promotion)  # 0..2
        plane = 64 + promo_idx * 3 + dir_idx
        return plane, r0, c0

    # 2) Knight planes 56-63
    for i, (kdr, kdc) in enumerate(KNIGHT_DELTAS):
        if dr == kdr and dc == kdc:
            return 56 + i, r0, c0

    # 3) Sliding planes 0-55
    # Must be straight or diagonal, and distance 1..7
    dist = max(abs(dr), abs(dc))
    if dist == 0 or dist > 7:
        raise ValueError("Invalid slide distance")

    # Direction must match one of SLIDE_DIRS times dist
    for dir_idx, (sdr, sdc) in enumerate(SLIDE_DIRS):
        if dr == sdr * dist and dc == sdc * dist:
            # plane index: dir_idx * 7 + (dist-1)
            plane = dir_idx * 7 + (dist - 1)
            return plane, r0, c0

    raise ValueError(f"Move delta not representable in 73-plane scheme: dr={dr}, dc={dc}")


def generate_realistic_white_uci_vocab():
    move_vocab = set()
    board = chess.Board()
    board.clear()

    white_pieces = [chess.KNIGHT, chess.QUEEN]

    for piece_type in white_pieces:
        for from_square in chess.SQUARES:

            board.clear()
            board.set_piece_at(from_square, chess.Piece(piece_type, chess.WHITE))
            board.turn = chess.WHITE

            for move in board.legal_moves:
                move_vocab.add(move.uci())

    return sorted(move_vocab)

def generate_realistic_black_uci_vocab():
    move_vocab = set()
    board = chess.Board()
    board.clear()

    black_pieces = [
        chess.PAWN,
        chess.KNIGHT,
        chess.QUEEN
    ]

    for piece_type in black_pieces:
        for from_square in chess.SQUARES:
            # Skip edge ranks for pawns (can’t be placed there)
            if piece_type == chess.PAWN and chess.square_rank(from_square) in [0, 7]:
                continue

            board.clear()
            board.set_piece_at(from_square, chess.Piece(piece_type, chess.BLACK))
            board.turn = chess.BLACK


            for move in board.legal_moves:
                move_vocab.add(move.uci())

    return sorted(move_vocab)

def generate_move_dictionary():
    move_dict = {}
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq != to_sq:
                move = chess.Move(from_square=from_sq, to_square=to_sq)
                try:
                    encoded = encode_move_plane_row_col(turn=chess.WHITE, move=move)
                    move_dict[move] = encoded
                except:
                    pass

    from_squares = ["a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"]
    to_squares = ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"]
    promos = ["q", "r", "b", "n"]
    for from_sq in from_squares:
        for to_sq in to_squares:
            for promo in promos:
                uci = from_sq + to_sq + promo
                move = uci_to_move(uci)
                try:
                    encoded = encode_move_plane_row_col(turn=chess.WHITE, move=move)
                    move_dict[move] = encoded
                except:
                    pass
    return move_dict

def main():
    move_dict = generate_move_dictionary()
    queen_promo = uci_to_move("g7h8n")
    regular_move = uci_to_move("e7e8")
    print(move_dict[queen_promo])
    print(move_dict[regular_move])
                
    


if __name__ == "__main__":
    main()



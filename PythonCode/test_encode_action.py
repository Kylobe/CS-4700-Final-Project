from ChessEnv import ChessEnv
import chess
import pytest




def assert_encoding_tuple(enc):
    assert isinstance(enc, tuple), f"Expected tuple, got {type(enc)}"
    assert len(enc) == 3, f"Expected (plane,row,col), got {enc}"
    p, r, c = enc
    assert isinstance(p, int) and isinstance(r, int) and isinstance(c, int), f"Non-int values: {enc}"
    assert 0 <= r < 8, f"Row out of bounds: {r} in {enc}"
    assert 0 <= c < 8, f"Col out of bounds: {c} in {enc}"
    assert p >= 0, f"Plane should be non-negative: {p} in {enc}"

def test_encode_action_is_deterministic():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    e1 = ChessEnv.encode_action(move, board)
    e2 = ChessEnv.encode_action(move, board)
    assert e1 == e2, "Encoding should be deterministic for same board+move"

def test_encode_action_row_col_matches_from_square():
    """
    Encodings use (row,col) as the FROM square (origin).
    """
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    p, r, c = ChessEnv.encode_action(move, board)
    assert_encoding_tuple((p, r, c))

    from_sq = move.from_square
    from_file = chess.square_file(from_sq)  # 0=a .. 7=h
    from_rank = chess.square_rank(from_sq)  # 0=rank1 .. 7=rank8

    # Common convention: row 0 is rank 1 OR rank 8 depending on implementation.
    # We'll accept either, but require consistency.
    assert c == from_file, f"Expected col==from_file ({from_file}), got {c}"

    # allow either "row = rank" or "row = 7-rank"
    assert r in (from_rank, 7 - from_rank), f"Row {r} doesn't match rank {from_rank} (or flipped)."

def test_encode_action_changes_when_origin_changes():
    board = chess.Board()
    m1 = chess.Move.from_uci("e2e4")
    m2 = chess.Move.from_uci("d2d4")
    e1 = ChessEnv.encode_action(m1, board)
    e2 = ChessEnv.encode_action(m2, board)
    assert e1[1:] != e2[1:], "If row/col represent origin square, different origins should differ"

def test_encode_action_same_relative_move_opposite_color_perspective():
    """
    If your encoding is 'from side-to-move perspective', then e2e4 (white)
    and d7d5 (black) often encode identically after white plays e2e4.
    This is exactly what your test is hinting at.
    """
    board = chess.Board()
    w = chess.Move.from_uci("e2e4")
    ew = ChessEnv.encode_action(w, board)
    board.push(w)

    b = chess.Move.from_uci("d7d5")
    eb = ChessEnv.encode_action(b, board)

    assert_encoding_tuple(ew)
    assert_encoding_tuple(eb)
    assert ew == eb, f"Expected symmetry encoding equality, got {ew} vs {eb}"

@pytest.mark.parametrize(
    "uci, expected",
    [
        ("e4d6", (57, 3, 4)),
        ("e4f6", (56, 3, 4)),
        ("e4g5", (60, 3, 4)),
        ("e4g3", (62, 3, 4)),
        ("e4f2", (58, 3, 4)),
        ("e4d2", (59, 3, 4)),
        ("e4c3", (63, 3, 4)),
        ("e4c5", (61, 3, 4)),
    ]
)
def test_encode_action_knight_move(uci, expected):
    board = chess.Board()
    move = chess.Move.from_uci(uci)
    enc = ChessEnv.encode_action(move, board)
    assert_encoding_tuple(enc)
    assert enc == expected

@pytest.mark.parametrize(
    "turn, uci, expected",
    [
        ("w", "e1g1", (15, 0, 4)),
        ("w", "e1c1", (22, 0, 4)),
        ("b", "e8g8", (22, 0, 3)),
        ("b", "e8c8", (15, 0, 3)),
    ]
)
def test_encode_action_castling(turn, uci, expected):
    board = chess.Board(f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1")
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    enc = ChessEnv.encode_action(move, board)
    assert_encoding_tuple(enc)
    assert enc == expected

@pytest.mark.parametrize(
    "turn, uci, expected",
    [
        ("w", "c7c8q", (0, 6, 2)),
        ("w", "c7b8q", (35, 6, 2)),
        ("w", "c7d8q", (28, 6, 2)),
        ("w", "c7c8n", (64, 6, 2)),
        ("w", "c7b8n", (65, 6, 2)),
        ("w", "c7d8n", (66, 6, 2)),
        ("w", "c7c8b", (67, 6, 2)),
        ("w", "c7b8b", (68, 6, 2)),
        ("w", "c7d8b", (69, 6, 2)),
        ("w", "c7c8r", (70, 6, 2)),
        ("w", "c7b8r", (71, 6, 2)),
        ("w", "c7d8r", (72, 6, 2)),

        ("b", "f2f1q", (0, 6, 2)),
        ("b", "f2g1q", (35, 6, 2)),
        ("b", "f2e1q", (28, 6, 2)),
        ("b", "f2f1n", (64, 6, 2)),
        ("b", "f2g1n", (65, 6, 2)),
        ("b", "f2e1n", (66, 6, 2)),
        ("b", "f2f1b", (67, 6, 2)),
        ("b", "f2g1b", (68, 6, 2)),
        ("b", "f2e1b", (69, 6, 2)),
        ("b", "f2f1r", (70, 6, 2)),
        ("b", "f2g1r", (71, 6, 2)),
        ("b", "f2e1r", (72, 6, 2)),
    ]
)
def test_encode_action_promotion_and_underpromotion(turn, uci, expected):
    board = chess.Board(f"8/P7/8/8/8/8/8/4k2K {turn} - - 0 1")
    move = chess.Move.from_uci(uci)
    enc = ChessEnv.encode_action(move, board)
    assert enc == expected

@pytest.mark.parametrize(
    "uci, expected",
    [
        ("c7c8q","f2f1q"),
        ("c3d3","f6e6"),
        ("c3e3","f6d6"),
        ("c3d4","f6e5"),
        ("c3b2","f6g7"),
        ("c3d5","f6e4"),
        ("c3b1","f6g8")
    ]
)
def test_mirror_move(uci, expected):
    input_move = chess.Move.from_uci(uci)
    expected_move = ChessEnv.mirror_move(input_move)
    assert expected == expected_move.uci()

def main():
    test_mirror_move("c7c8q","f2f1q")

if __name__ == "__main__":
    main()




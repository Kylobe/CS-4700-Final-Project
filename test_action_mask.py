from ChessEnv import ChessEnv
import chess
import pytest
import numpy as np


def test_starting_mask():
    board = chess.Board()
    mask = ChessEnv.create_plane_action_mask(board)

    Z = [[0]*8 for _ in range(8)]
    O = [[1]*8 for _ in range(8)]

    expected_mask = np.array(
        [
            [   #N Direction (Distance 1)
                Z[0],
                O[0],
                *Z[2:8]
            ],
            [   #N Direction (Distance 2)
                Z[0],
                O[0],
                *Z[2:8]
            ],
            [   #N Direction (Distance 3)
                *Z[0:8]
            ],
            [   #N Direction (Distance 4)
                *Z[0:8]
            ],
            [   #N Direction (Distance 5)
                *Z[0:8]
            ],
            [   #N Direction (Distance 6)
                *Z[0:8]
            ],
            [   #N Direction (Distance 7)
                *Z[0:8]
            ],
            [   #S Direction (Distance 1)
                *Z[0:8]
            ],
            [   #S Direction (Distance 2)
                *Z[0:8]
            ],
            [   #S Direction (Distance 3)
                *Z[0:8]
            ],
            [   #S Direction (Distance 4)
                *Z[0:8]
            ],
            [   #S Direction (Distance 5)
                *Z[0:8]
            ],
            [   #S Direction (Distance 6)
                *Z[0:8]
            ],
            [   #S Direction (Distance 7)
                *Z[0:8]
            ],
            [   #R Direction (Distance 1)
                *Z[0:8]
            ],
            [   #R Direction (Distance 2)
                *Z[0:8]
            ],
            [   #R Direction (Distance 3)
                *Z[0:8]
            ],
            [   #R Direction (Distance 4)
                *Z[0:8]
            ],
            [   #R Direction (Distance 5)
                *Z[0:8]
            ],
            [   #R Direction (Distance 6)
                *Z[0:8]
            ],
            [   #R Direction (Distance 7)
                *Z[0:8]
            ],
            [   #L Direction (Distance 1)
                *Z[0:8]
            ],
            [   #L Direction (Distance 2)
                *Z[0:8]
            ],
            [   #L Direction (Distance 3)
                *Z[0:8]
            ],
            [   #L Direction (Distance 4)
                *Z[0:8]
            ],
            [   #L Direction (Distance 5)
                *Z[0:8]
            ],
            [   #L Direction (Distance 6)
                *Z[0:8]
            ],
            [   #L Direction (Distance 7)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 1)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 2)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 3)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 4)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 5)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 6)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 7)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 1)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 2)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 3)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 4)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 5)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 6)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 7)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 1)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 2)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 3)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 4)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 5)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 6)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 7)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 1)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 2)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 3)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 4)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 5)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 6)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 7)
                *Z[0:8]
            ],
            [   #2NR Knight Move
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #2NL Knight Move
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #2SR Knight Move
                *Z[0:8]
            ],
            [   #2SL Knight Move
                *Z[0:8]
            ],
            [   #N2R Knight Move
                *Z[0:8]
            ],
            [   #N2L Knight Move
                *Z[0:8]
            ],
            [   #S2R Knight Move
                *Z[0:8]
            ],
            [   #S2L Knight Move
                *Z[0:8]
            ],
            [   #Promo To Knight N
                *Z[0:8]
            ],
            [   #Promo To Knight L
                *Z[0:8]
            ],
            [   #Promo To Knight R
                *Z[0:8]
            ],
            [   #Promo To Bishop N
                *Z[0:8]
            ],
            [   #Promo To Bishop L
                *Z[0:8]
            ],
            [   #Promo To Bishop R
                *Z[0:8]
            ],
            [   #Promo To Rook N
                *Z[0:8]
            ],
            [   #Promo To Rook L
                *Z[0:8]
            ],
            [   #Promo To Rook R
                *Z[0:8]
            ],
        ],
        dtype=np.int8
        )
    assert np.array_equal(expected_mask, mask)

def test_flipped_board():
    board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1")
    mask = ChessEnv.create_plane_action_mask(board)

    Z = [[0]*8 for _ in range(8)]
    O = [[1]*8 for _ in range(8)]

    expected_mask = np.array(
        [
            [   #N Direction (Distance 1)
                [0,0,0,1,1,0,0,0],
                [1,1,1,0,0,1,1,1],
                *Z[2:8]
            ],
            [   #N Direction (Distance 2)
                [0,0,0,0,1,0,0,0],
                [1,1,1,0,0,1,1,1],
                *Z[2:8]
            ],
            [   #N Direction (Distance 3)
                *Z[0:8]
            ],
            [   #N Direction (Distance 4)
                *Z[0:8]
            ],
            [   #N Direction (Distance 5)
                *Z[0:8]
            ],
            [   #N Direction (Distance 6)
                *Z[0:8]
            ],
            [   #N Direction (Distance 7)
                *Z[0:8]
            ],
            [   #S Direction (Distance 1)
                *Z[0:8]
            ],
            [   #S Direction (Distance 2)
                *Z[0:8]
            ],
            [   #S Direction (Distance 3)
                *Z[0:8]
            ],
            [   #S Direction (Distance 4)
                *Z[0:8]
            ],
            [   #S Direction (Distance 5)
                *Z[0:8]
            ],
            [   #S Direction (Distance 6)
                *Z[0:8]
            ],
            [   #S Direction (Distance 7)
                *Z[0:8]
            ],
            [   #R Direction (Distance 1)
                *Z[0:8]
            ],
            [   #R Direction (Distance 2)
                *Z[0:8]
            ],
            [   #R Direction (Distance 3)
                *Z[0:8]
            ],
            [   #R Direction (Distance 4)
                *Z[0:8]
            ],
            [   #R Direction (Distance 5)
                *Z[0:8]
            ],
            [   #R Direction (Distance 6)
                *Z[0:8]
            ],
            [   #R Direction (Distance 7)
                *Z[0:8]
            ],
            [   #L Direction (Distance 1)
                *Z[0:8]
            ],
            [   #L Direction (Distance 2)
                *Z[0:8]
            ],
            [   #L Direction (Distance 3)
                *Z[0:8]
            ],
            [   #L Direction (Distance 4)
                *Z[0:8]
            ],
            [   #L Direction (Distance 5)
                *Z[0:8]
            ],
            [   #L Direction (Distance 6)
                *Z[0:8]
            ],
            [   #L Direction (Distance 7)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 1)
                [0,0,1,1,0,0,0,0],
                Z[0],
                Z[0],
                [0,0,0,1,0,0,0,0],
                *Z[4:8]
            ],
            [   #NR Direction (Distance 2)
                [0,0,1,0,0,0,0,0],
                *Z[1:8]
            ],
            [   #NR Direction (Distance 3)
                [0,0,1,0,0,0,0,0],
                *Z[1:8]
            ],
            [   #NR Direction (Distance 4)
                [0,0,1,0,0,0,0,0],
                *Z[1:8]
            ],
            [   #NR Direction (Distance 5)
                [0,0,1,0,0,0,0,0],
                *Z[1:8]
            ],
            [   #NR Direction (Distance 6)
                *Z[0:8]
            ],
            [   #NR Direction (Distance 7)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 1)
                [0,0,0,0,1,1,0,0],
                Z[0],
                Z[0],
                [0,0,0,0,1,0,0,0],
                *Z[4:8]
            ],
            [   #NL Direction (Distance 2)
                [0,0,0,0,1,1,0,0],
                *Z[1:8]
            ],
            [   #NL Direction (Distance 3)
                [0,0,0,0,1,1,0,0],
                *Z[1:8]
            ],
            [   #NL Direction (Distance 4)
                [0,0,0,0,1,1,0,0],
                *Z[1:8]
            ],
            [   #NL Direction (Distance 5)
                [0,0,0,0,0,1,0,0],
                *Z[1:8]
            ],
            [   #NL Direction (Distance 6)
                *Z[0:8]
            ],
            [   #NL Direction (Distance 7)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 1)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 2)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 3)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 4)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 5)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 6)
                *Z[0:8]
            ],
            [   #SR Direction (Distance 7)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 1)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 2)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 3)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 4)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 5)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 6)
                *Z[0:8]
            ],
            [   #SL Direction (Distance 7)
                *Z[0:8]
            ],
            [   #2NR Knight Move
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #2NL Knight Move
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #2SR Knight Move
                *Z[0:8]
            ],
            [   #2SL Knight Move
                *Z[0:8]
            ],
            [   #N2R Knight Move
                [0,1,0,0,0,0,0,0],
                *Z[1:8]
            ],
            [   #N2L Knight Move
                [0,0,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #S2R Knight Move
                *Z[0:8]
            ],
            [   #S2L Knight Move
                *Z[0:8]
            ],
            [   #Promo To Knight N
                *Z[0:8]
            ],
            [   #Promo To Knight L
                *Z[0:8]
            ],
            [   #Promo To Knight R
                *Z[0:8]
            ],
            [   #Promo To Bishop N
                *Z[0:8]
            ],
            [   #Promo To Bishop L
                *Z[0:8]
            ],
            [   #Promo To Bishop R
                *Z[0:8]
            ],
            [   #Promo To Rook N
                *Z[0:8]
            ],
            [   #Promo To Rook L
                *Z[0:8]
            ],
            [   #Promo To Rook R
                *Z[0:8]
            ],
        ],
        dtype=np.int8
        )
    if not np.array_equal(mask, expected_mask):
        diff = np.where(mask != expected_mask)
        assert False, f"Mismatch at planes, rows, cols: {diff}"


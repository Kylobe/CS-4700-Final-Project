from ChessEnv import ChessEnv
import chess
import pytest
import torch
import numpy as np

def test_starting_board():
    board = chess.Board()
    enc = ChessEnv.encode_board(board)

    Z = [[0]*8 for _ in range(8)]
    O = [[1]*8 for _ in range(8)]

    np_piece_planes = np.array(
        [
            [   #W Pawns
                Z[0],
                [1,1,1,1,1,1,1,1],
                *Z[2:8],
            ],
            [   #W Knights
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #W Bishops
                [0,0,1,0,0,1,0,0],
                *Z[1:8]
            ],
            [   #W Rooks
                [1,0,0,0,0,0,0,1],
                *Z[1:8]
            ],
            [   #W Queens
                [0,0,0,1,0,0,0,0],
                *Z[1:8]
            ],
            [   #W Kings
                [0,0,0,0,1,0,0,0],
                *Z[1:8]
            ],
            [   #B Pawns
                *Z[0:6],
                [1,1,1,1,1,1,1,1],
                Z[7]
            ],
            [   #B Knights
                *Z[0:7],
                [0,1,0,0,0,0,1,0]
            ],
            [   #B Bishops
                *Z[0:7],
                [0,0,1,0,0,1,0,0]
            ],
            [   #B Rooks
                *Z[0:7],
                [1,0,0,0,0,0,0,1]
            ],
            [   #B Queens
                *Z[0:7],
                [0,0,0,1,0,0,0,0]
            ],
            [   #B Kings
                *Z[0:7],
                [0,0,0,0,1,0,0,0]
            ],
            [   #W King Side Castling
                *O[0:8]
            ],
            [   #W Queen Side Castling
                *O[0:8]
            ],
            [   #B King Side Castling
                *O[0:8]
            ],
            [   #B Queen Side Castling
                *O[0:8]
            ],
            [   #En Pessant
                *Z[0:8]
            ]
        ],
        dtype=np.int8
        )
    expected = torch.from_numpy(np_piece_planes)
    expected = expected.to(dtype=enc.dtype, device=enc.device)
    assert torch.equal(enc, expected)

def test_black_on_the_play():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    enc = ChessEnv.encode_board(board)

    Z = [[0]*8 for _ in range(8)]
    O = [[1]*8 for _ in range(8)]

    np_piece_planes = np.array(
        [
            [   #B Pawns
                Z[0],
                [1,1,1,1,1,1,1,1],
                *Z[2:8]
            ],
            [   #B Knights
                [0,1,0,0,0,0,1,0],
                *Z[1:8]
            ],
            [   #B Bishops
                [0,0,1,0,0,1,0,0],
                *Z[1:8]
            ],
            [   #B Rooks
                [1,0,0,0,0,0,0,1],
                *Z[1:8]
            ],
            [   #B Queens
                [0,0,0,0,1,0,0,0],
                *Z[1:8]
            ],
            [   #B Kings
                [0,0,0,1,0,0,0,0],
                *Z[1:8]
            ],
            [   #W Pawns
                *Z[0:6],
                [1,1,1,1,1,1,1,1],
                Z[0]
            ],
            [   #W Knights
                *Z[0:7],
                [0,1,0,0,0,0,1,0]
            ],
            [   #W Bishops
                *Z[0:7],
                [0,0,1,0,0,1,0,0]
            ],
            [   #W Rooks
                *Z[0:7],
                [1,0,0,0,0,0,0,1]
            ],
            [   #W Queens
                *Z[0:7],
                [0,0,0,0,1,0,0,0]
            ],
            [   #W Kings
                *Z[0:7],
                [0,0,0,1,0,0,0,0]
            ],
            [   #B King Side Castling
                *O[0:8]
            ],
            [   #B Queen Side Castling
                *O[0:8]
            ],
            [   #W King Side Castling
                *O[0:8]
            ],
            [   #W Queen Side Castling
                *O[0:8]
            ],
            [   #En Pessant
                *Z[0:8]
            ]
        ],
        dtype=np.int8
        )
    expected = torch.from_numpy(np_piece_planes)
    expected = expected.to(dtype=enc.dtype, device=enc.device)
    assert torch.equal(enc, expected)

def test_en_pessant():
    board = chess.Board("4k3/8/8/1Pp5/8/8/8/4K3 w - c6 0 1")
    enc = ChessEnv.encode_board(board)

    Z = [[0]*8 for _ in range(8)]
    O = [[1]*8 for _ in range(8)]

    np_piece_planes = np.array(
        [
            [   #W Pawns
                *Z[0:4],
                [0,1,0,0,0,0,0,0],
                *Z[5:8]
            ],
            [   #W Knights
                *Z[0:8]
            ],
            [   #W Bishops
                *Z[0:8]
            ],
            [   #W Rooks
                *Z[0:8]
            ],
            [   #W Queens
                *Z[0:8]
            ],
            [   #W Kings
                [0,0,0,0,1,0,0,0],
                *Z[1:8]
            ],
            [   #B Pawns
                *Z[0:4],
                [0,0,1,0,0,0,0,0],
                *Z[5:8]
            ],
            [   #B Knights
                *Z[0:8]
            ],
            [   #B Bishops
                *Z[0:8]
            ],
            [   #B Rooks
                *Z[0:8]
            ],
            [   #B Queens
                *Z[0:8]
            ],
            [   #B Kings
                *Z[0:7],
                [0,0,0,0,1,0,0,0]
            ],
            [   #W King Side Castling
                *Z[0:8]
            ],
            [   #W Queen Side Castling
                *Z[0:8]
            ],
            [   #B King Side Castling
                *Z[0:8]
            ],
            [   #B Queen Side Castling
                *Z[0:8]
            ],
            [   #En Pessant
                *Z[0:5],
                [0,0,1,0,0,0,0,0],
                *Z[6:8]
            ]
        ],
        dtype=np.int8
        )
    expected = torch.from_numpy(np_piece_planes)
    expected = expected.to(dtype=enc.dtype, device=enc.device)
    assert torch.equal(enc, expected), f"Mismatch at planes, rows, cols: {torch.where(expected != enc)}"


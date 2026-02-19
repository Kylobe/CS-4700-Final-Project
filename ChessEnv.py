import gymnasium as gym
from gymnasium import spaces
import numpy as np
import chess
import torch
import chess.engine
import random
from legal_uci import generate_move_dictionary

class ChessEnv(gym.Env):
    move_to_encoding = generate_move_dictionary()
    encoding_to_move = {encoding: move for move, encoding in move_to_encoding.items() if move.promotion != chess.QUEEN}
    def __init__(self):
        self.board = chess.Board()
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 17), dtype=np.float32)
        
        self.num_moves = len(self.move_to_encoding)
        self.action_space = spaces.Discrete(self.num_moves)
        self.piece_vals = {
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }

        with open("endgame_fens.txt", mode='r', encoding='utf-8') as file:
            self.endgame_fens = [line.strip() for line in file]

    def reset(self, is_endgame=False):
        self.board.reset()
        if is_endgame and self.endgame_fens:
            fen = random.choice(self.endgame_fens)
            try:
                self.board.set_fen(fen)
            except ValueError:
                print(f"[Warning] Invalid FEN: {fen}, falling back to default position.")
                self.board.reset()
        return self.board.copy(stack=False)

    def step(self, action):
        move = ChessEnv.decode_action(action, self.board)

        if move not in self.board.legal_moves:
            print("Illegal move chosen")
            return self._get_obs(), True

        self.board.push(move)
        done = self.board.is_game_over()
        if self.board.can_claim_fifty_moves() or self.board.can_claim_threefold_repetition():
            done = True

        return self.board.copy(stack=False), done

    def render(self, mode='human'):
        print(self.board)

    def _get_obs(self):
        return ChessEnv.encode_board(self.board)

    @staticmethod
    def decode_action(encoding: int, board: chess.Board):
        base = ChessEnv.encoding_to_move[encoding]
        move = chess.Move(base.from_square, base.to_square, promotion=base.promotion)

        if board.turn == chess.BLACK:
            move = ChessEnv.mirror_move(move)

        if move not in board.legal_moves:
            # fallback promotion behavior
            move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        return move

    @staticmethod
    def encode_action(base: chess.Move, board: chess.Board):
        move = chess.Move(base.from_square, base.to_square, promotion=base.promotion)
        if board.turn == chess.BLACK:
            move = ChessEnv.mirror_move(move)
        if move.promotion == chess.QUEEN:
            move = chess.Move(move.from_square, move.to_square, promotion=None)
        return ChessEnv.move_to_encoding[move]

    @staticmethod
    def create_plane_action_mask(board: chess.Board, n_planes=73):
        mask = np.zeros((n_planes, 8, 8), dtype=np.float32)
        for mv in board.legal_moves:
            mask[ChessEnv.encode_action(mv, board)] = 1.0
        return mask

    @staticmethod
    def encode_board(board:chess.Board):
        piece_planes = torch.zeros((12, 8, 8), dtype=torch.float32)

        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            piece_type = piece.piece_type - 1
            color_offset = 0 if piece.color == board.turn else 6
            row, col = divmod(square, 8)
            piece_planes[piece_type + color_offset][row][col] = 1.0

        enemy = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
        castling = torch.zeros((4, 8, 8))
        if board.has_kingside_castling_rights(board.turn): castling[0].fill_(1.0)
        if board.has_queenside_castling_rights(board.turn): castling[1].fill_(1.0)
        if board.has_kingside_castling_rights(enemy): castling[2].fill_(1.0)
        if board.has_queenside_castling_rights(enemy): castling[3].fill_(1.0)

        en_passant = torch.zeros((1, 8, 8), dtype=torch.float32)
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)  
            en_passant[0, row, col] = 1.0
        
        my_tensor = torch.cat([piece_planes, castling, en_passant], dim=0)
        if board.turn == chess.BLACK:
            my_tensor = torch.flip(my_tensor, [1, 2])

        return my_tensor

    @staticmethod
    def mirror_move(move: chess.Move):
        cur_from_row, cur_from_col = divmod(move.from_square, 8)
        cur_to_row, cur_to_col = divmod(move.to_square, 8)
        new_from_row = -cur_from_row + 7
        new_from_col = -cur_from_col + 7
        new_to_row =  -cur_to_row + 7
        new_to_col = -cur_to_col + 7
        new_from_square = new_from_row * 8 + new_from_col
        new_to_square = new_to_row * 8 + new_to_col
        new_move = chess.Move(new_from_square, new_to_square, promotion=move.promotion)
        return new_move

    def peek_at_next_state(self, action):
        new_board = self.board.copy(stack=False)
        new_board.push(action)
        return self._encode_board(new_board)

    def material_advantage(self, board:chess.Board):
        score = 0
        for piece_type, value in self.piece_vals.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        # from side-to-move perspective
        return score if board.turn == chess.WHITE else -score



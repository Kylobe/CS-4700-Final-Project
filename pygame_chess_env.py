import sys
import time
import threading
import queue
import pygame
import chess
from MCTS import MCTS
from AlphaZeroChess import AlphaZeroChess
import torch
import numpy as np
from ChessEnv import ChessEnv
import chess.engine


# -----------------------------
# Config
# -----------------------------
SQ_SIZE = 80
BOARD_SIZE = SQ_SIZE * 8
PANEL_W = 260
W, H = BOARD_SIZE + PANEL_W, BOARD_SIZE
FPS = 60

LIGHT = (240, 217, 181)
DARK  = (181, 136,  99)
HILITE_SEL   = (120, 170, 255)
HILITE_MOVE  = (120, 220, 140)
HILITE_CAP   = (255, 140, 140)
HILITE_LAST  = (255, 225, 120)

TEXT = (20, 20, 20)
BG_PANEL = (245, 245, 245)
LINE = (210, 210, 210)

UNICODE_PIECE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# -----------------------------
# Bot interface (YOU IMPLEMENT)
# -----------------------------
class MCTSBot:
    """
    Implement choose_move(board) to return a python-chess Move.
    board is a python-chess Board in the current position (Black to move).
    """

    def __init__(self, max_think_time_s: float):
        args = {
            'C': 2,
            'num_searches': 1600,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'res_blocks': 40,
            'num_hidden': 256,
        }

        # Your trained model
        model = AlphaZeroChess(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])

        model.load_state_dict(torch.load("PretrainModel.pt", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        model.eval()
        self.mcts = MCTS(args=args, model=model)
        self.max_think_time_s = max_think_time_s

    def choose_move(self, board: chess.Board) -> chess.Move:
        action_probs = self.mcts.search(board)
        flat_index = np.argmax(action_probs)
        action = np.unravel_index(flat_index, action_probs.shape)
        return ChessEnv.decode_action(action, board)

    def update_root(self, action):
        self.mcts.advance_root(action)

    def reset_root(self):
        self.mcts.root = None

    def create_root(self, board):
        self.mcts.create_root(board)

class StockFishBot:
    def __init__(self, max_think_time_s: float):
        self.engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Traedon Harris\Documents\GitHub\CS-4700-Final-Project\stockfish\stockfish-windows-x86-64-avx2.exe")
        self.max_think_time_s = max_think_time_s


    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board,
            chess.engine.Limit(depth=5)
        )
        return result.move

    def update_root(self, action):
        pass

    def reset_root(self):
        pass

    def create_root(self, board):
        pass


# -----------------------------
# Helpers
# -----------------------------
def square_to_xy(square: chess.Square) -> tuple[int, int]:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    x = file * SQ_SIZE
    y = (7 - rank) * SQ_SIZE
    return x, y

def xy_to_square(mx: int, my: int):
    if mx < 0 or my < 0 or mx >= BOARD_SIZE or my >= BOARD_SIZE:
        return None
    file = mx // SQ_SIZE
    rank_from_top = my // SQ_SIZE
    rank = 7 - rank_from_top
    return chess.square(file, rank)

def draw_board(screen, selected_sq, legal_to_sqs, capture_to_sqs, last_move):
    for rank in range(8):
        for file in range(8):
            x = file * SQ_SIZE
            y = rank * SQ_SIZE
            is_light = (file + rank) % 2 == 0
            color = LIGHT if is_light else DARK
            screen.fill(color, (x, y, SQ_SIZE, SQ_SIZE))

    if last_move is not None:
        for sq in [last_move.from_square, last_move.to_square]:
            x, y = square_to_xy(sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((*HILITE_LAST, 120))
            screen.blit(s, (x, y))

    if selected_sq is not None:
        x, y = square_to_xy(selected_sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_SEL, 130))
        screen.blit(s, (x, y))

    for sq in legal_to_sqs:
        x, y = square_to_xy(sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_MOVE, 110))
        screen.blit(s, (x, y))

    for sq in capture_to_sqs:
        x, y = square_to_xy(sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_CAP, 130))
        screen.blit(s, (x, y))

    for i in range(9):
        pygame.draw.line(screen, (0, 0, 0), (i * SQ_SIZE, 0), (i * SQ_SIZE, BOARD_SIZE), 1)
        pygame.draw.line(screen, (0, 0, 0), (0, i * SQ_SIZE), (BOARD_SIZE, i * SQ_SIZE), 1)

def draw_pieces(screen, board, piece_font):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            sym = UNICODE_PIECE[piece.symbol()]
            x, y = square_to_xy(square)
            surf = piece_font.render(sym, True, (0, 0, 0))
            rect = surf.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2 + 2))
            screen.blit(surf, rect)

def draw_panel(screen, board, ui_font, small_font, status_msg, bot_thinking: bool):
    screen.fill(BG_PANEL, (BOARD_SIZE, 0, PANEL_W, H))
    pygame.draw.line(screen, LINE, (BOARD_SIZE, 0), (BOARD_SIZE, H), 2)

    x0 = BOARD_SIZE + 14
    y = 14

    def line(text, font=ui_font, pad=10):
        nonlocal y
        surf = font.render(text, True, TEXT)
        screen.blit(surf, (x0, y))
        y += surf.get_height() + pad

    line("Chess (Human vs MCTS)")
    line(f"Turn: {'White' if board.turn else 'Black'}", pad=6)

    if board.is_checkmate():
        line("Checkmate!", pad=4)
        line(f"Winner: {'Black' if board.turn else 'White'}", pad=10)
    elif board.is_stalemate():
        line("Stalemate!", pad=10)
    elif board.is_insufficient_material():
        line("Draw (insufficient)", pad=10)
    elif board.is_check():
        line("Check!", pad=10)

    if bot_thinking:
        line("Bot: thinking...", pad=10)

    line("Controls:", pad=6)
    line("• You are White", small_font, pad=6)
    line("• Click piece, then target", small_font, pad=6)
    line("• R = reset", small_font, pad=6)
    line("• U = undo (2 plies)", small_font, pad=12)

    if status_msg:
        y += 6
        for part in status_msg.split("\n"):
            surf = small_font.render(part, True, (60, 60, 60))
            screen.blit(surf, (x0, y))
            y += surf.get_height() + 4

def choose_promotion_piece(board, move: chess.Move) -> chess.PieceType:
    return chess.QUEEN


# -----------------------------
# Bot threading (keeps UI alive)
# -----------------------------
def bot_worker(req_q: "queue.Queue[str]", res_q: "queue.Queue[chess.Move]", bot: MCTSBot, board_ref):
    """
    Wait for 'go' messages; compute a move from a *copy* of the board to avoid races.
    """
    while True:
        msg = req_q.get()
        if msg is None:
            return
        # Copy board so bot can think without UI thread mutating it
        board_copy = board_ref.copy(stack=False)
        mv = bot.choose_move(board_copy)
        res_q.put(mv)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pygame Chess - Human vs MCTS")
    clock = pygame.time.Clock()

    piece_font = pygame.font.SysFont("Segoe UI Symbol", 56) or pygame.font.Font(None, 56)
    ui_font = pygame.font.SysFont("Segoe UI", 24) or pygame.font.Font(None, 24)
    small_font = pygame.font.SysFont("Segoe UI", 18) or pygame.font.Font(None, 18)

    board = chess.Board()

    # Human is White
    HUMAN_COLOR = chess.WHITE
    BOT_COLOR = chess.BLACK

    bot = StockFishBot(max_think_time_s=300)
    bot.create_root(board)

    selected_sq = None
    legal_moves_from_selected = []
    legal_to_sqs = set()
    capture_to_sqs = set()
    last_move = None
    status_msg = "You are White. Make a move."

    bot_thinking = False
    bot_req_q: "queue.Queue[str]" = queue.Queue()
    bot_res_q: "queue.Queue[chess.Move]" = queue.Queue()
    t = threading.Thread(target=bot_worker, args=(bot_req_q, bot_res_q, bot, board), daemon=True)
    t.start()

    def clear_selection():
        nonlocal selected_sq, legal_moves_from_selected, legal_to_sqs, capture_to_sqs
        selected_sq = None
        legal_moves_from_selected = []
        legal_to_sqs.clear()
        capture_to_sqs.clear()

    def recompute_legal(selected):
        nonlocal legal_moves_from_selected, legal_to_sqs, capture_to_sqs
        legal_moves_from_selected = []
        legal_to_sqs = set()
        capture_to_sqs = set()

        if selected is None:
            return
        piece = board.piece_at(selected)
        if piece is None:
            return

        # Only allow selecting YOUR pieces, and only on YOUR turn
        if board.turn != HUMAN_COLOR:
            return
        if piece.color != HUMAN_COLOR:
            return

        for mv in board.legal_moves:
            if mv.from_square == selected:
                legal_moves_from_selected.append(mv)
                legal_to_sqs.add(mv.to_square)
                if board.is_capture(mv):
                    capture_to_sqs.add(mv.to_square)

    def try_make_human_move(from_sq, to_sq) -> bool:
        nonlocal last_move, status_msg
        if board.turn != HUMAN_COLOR:
            return False

        candidate_moves = [mv for mv in legal_moves_from_selected
                           if mv.to_square == to_sq and mv.from_square == from_sq]
        if not candidate_moves:
            return False

        mv = None
        if len(candidate_moves) == 1:
            mv = candidate_moves[0]
        else:
            promo = choose_promotion_piece(board, candidate_moves[0])
            for cm in candidate_moves:
                if cm.promotion == promo:
                    mv = cm
                    break
            mv = mv or candidate_moves[0]

        encoded_action = ChessEnv.encode_action(mv, board)

        bot.update_root(encoded_action)

        board.push(mv)
        last_move = mv
        status_msg = f"You played: {mv.uci()}"
        return True

    def start_bot_turn_if_needed():
        nonlocal bot_thinking, status_msg
        if board.is_game_over():
            return
        if board.turn == BOT_COLOR and not bot_thinking:
            bot_thinking = True
            status_msg = "Bot thinking..."
            bot_req_q.put("go")

    def apply_bot_move_if_ready():
        nonlocal bot_thinking, last_move, status_msg
        if not bot_thinking:
            return
        try:
            mv = bot_res_q.get_nowait()
        except queue.Empty:
            return

        bot_thinking = False

        # Safety: ensure move is legal in current board
        if mv not in board.legal_moves:
            # Fallback if bot returned something stale/illegal
            mv = next(iter(board.legal_moves))
            status_msg = "Bot returned illegal move; used fallback."
        else:
            status_msg = f"Bot played: {mv.uci()}"

        encoded_move = ChessEnv.encode_action(mv, board)
        bot.update_root(encoded_move)

        board.push(mv)
        last_move = mv

    # Kick off bot if Black starts (it won’t, since White starts)
    start_bot_turn_if_needed()

    running = True
    while running:
        clock.tick(FPS)

        # If bot finished thinking, apply its move
        apply_bot_move_if_ready()

        # If it's bot's turn, ensure bot is running
        start_bot_turn_if_needed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board.reset()
                    last_move = None
                    bot_thinking = False
                    while not bot_req_q.empty():
                        bot_req_q.get_nowait()
                    while not bot_res_q.empty():
                        bot_res_q.get_nowait()
                    clear_selection()
                    status_msg = "Game reset. You are White."
                    bot.reset_root()
                    bot.create_root(board)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if bot_thinking:
                    continue
                if board.is_game_over():
                    continue
                if board.turn != HUMAN_COLOR:
                    continue  # ignore clicks on bot's turn

                mx, my = pygame.mouse.get_pos()
                sq = xy_to_square(mx, my)
                if sq is None:
                    continue

                if selected_sq is None:
                    piece = board.piece_at(sq)
                    if piece and piece.color == HUMAN_COLOR:
                        selected_sq = sq
                        recompute_legal(selected_sq)
                        status_msg = "Select a destination square." if legal_to_sqs else "No legal moves."
                    else:
                        status_msg = "Select one of your (White) pieces."
                else:
                    moved = try_make_human_move(selected_sq, sq)
                    clear_selection()
                    if not moved:
                        # allow reselect
                        piece = board.piece_at(sq)
                        if piece and piece.color == HUMAN_COLOR:
                            selected_sq = sq
                            recompute_legal(selected_sq)
                            status_msg = "Switched selection."
                        else:
                            status_msg = "Illegal move."

        # Draw
        draw_board(screen, selected_sq, legal_to_sqs, capture_to_sqs, last_move)
        draw_pieces(screen, board, piece_font)
        draw_panel(screen, board, ui_font, small_font, status_msg, bot_thinking)
        pygame.display.flip()

    # stop worker thread
    bot_req_q.put(None)
    pygame.quit()
    sys.exit()
    bot.engine.quit()



if __name__ == "__main__":
    main()

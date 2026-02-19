import sys
import pygame
import chess

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

# Unicode chess pieces (works well in many fonts)
UNICODE_PIECE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# -----------------------------
# Helpers
# -----------------------------
def square_to_xy(square: chess.Square) -> tuple[int, int]:
    """Map python-chess square -> top-left pixel (x,y) with White at bottom."""
    file = chess.square_file(square)  # 0..7 (a..h)
    rank = chess.square_rank(square)  # 0..7 (1..8)
    x = file * SQ_SIZE
    y = (7 - rank) * SQ_SIZE
    return x, y

def xy_to_square(mx: int, my: int) -> chess.Square | None:
    """Map mouse pixel -> python-chess square, or None if outside board."""
    if mx < 0 or my < 0 or mx >= BOARD_SIZE or my >= BOARD_SIZE:
        return None
    file = mx // SQ_SIZE
    rank_from_top = my // SQ_SIZE
    rank = 7 - rank_from_top
    return chess.square(file, rank)

def draw_board(screen, selected_sq, legal_to_sqs, capture_to_sqs, last_move):
    # Draw squares
    for rank in range(8):
        for file in range(8):
            x = file * SQ_SIZE
            y = rank * SQ_SIZE
            is_light = (file + rank) % 2 == 0
            color = LIGHT if is_light else DARK
            screen.fill(color, (x, y, SQ_SIZE, SQ_SIZE))

    # Highlight last move
    if last_move is not None:
        for sq in [last_move.from_square, last_move.to_square]:
            x, y = square_to_xy(sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((*HILITE_LAST, 120))
            screen.blit(s, (x, y))

    # Highlight selected square
    if selected_sq is not None:
        x, y = square_to_xy(selected_sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_SEL, 130))
        screen.blit(s, (x, y))

    # Highlight legal moves
    for sq in legal_to_sqs:
        x, y = square_to_xy(sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_MOVE, 110))
        screen.blit(s, (x, y))

    # Highlight captures a bit differently
    for sq in capture_to_sqs:
        x, y = square_to_xy(sq)
        s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        s.fill((*HILITE_CAP, 130))
        screen.blit(s, (x, y))

    # Grid lines
    for i in range(9):
        pygame.draw.line(screen, (0, 0, 0), (i * SQ_SIZE, 0), (i * SQ_SIZE, BOARD_SIZE), 1)
        pygame.draw.line(screen, (0, 0, 0), (0, i * SQ_SIZE), (BOARD_SIZE, i * SQ_SIZE), 1)

def draw_pieces(screen, board, piece_font):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            sym = UNICODE_PIECE[piece.symbol()]
            x, y = square_to_xy(square)
            # Center the glyph in the square
            surf = piece_font.render(sym, True, (0, 0, 0))
            rect = surf.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2 + 2))
            screen.blit(surf, rect)

def draw_panel(screen, board, ui_font, small_font, status_msg):
    # Panel background
    screen.fill(BG_PANEL, (BOARD_SIZE, 0, PANEL_W, H))
    pygame.draw.line(screen, LINE, (BOARD_SIZE, 0), (BOARD_SIZE, H), 2)

    x0 = BOARD_SIZE + 14
    y = 14

    def line(text, font=ui_font, pad=10):
        nonlocal y
        surf = font.render(text, True, TEXT)
        screen.blit(surf, (x0, y))
        y += surf.get_height() + pad

    line("Chess (Pygame)")
    line(f"Turn: {'White' if board.turn else 'Black'}", pad=6)

    # Game state
    if board.is_checkmate():
        line("Checkmate!", pad=4)
        line(f"Winner: {'Black' if board.turn else 'White'}", pad=10)
    elif board.is_stalemate():
        line("Stalemate!", pad=10)
    elif board.is_insufficient_material():
        line("Draw (insufficient)", pad=10)
    elif board.is_check():
        line("Check!", pad=10)

    line("Controls:", pad=6)
    line("• Click piece, then click target", small_font, pad=6)
    line("• R = reset game", small_font, pad=6)
    line("• U = undo move", small_font, pad=12)

    if status_msg:
        # Wrap-ish: just render multiple lines manually
        y += 6
        for part in status_msg.split("\n"):
            surf = small_font.render(part, True, (60, 60, 60))
            screen.blit(surf, (x0, y))
            y += surf.get_height() + 4

def choose_promotion_piece(board, move: chess.Move) -> chess.PieceType:
    """
    Minimal promotion logic: auto-queen.
    You can expand this to UI selection.
    """
    return chess.QUEEN

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pygame Chess")
    clock = pygame.time.Clock()

    # Fonts: try common fonts; fallback to default
    piece_font = pygame.font.SysFont("Segoe UI Symbol", 56) or pygame.font.Font(None, 56)
    ui_font = pygame.font.SysFont("Segoe UI", 24) or pygame.font.Font(None, 24)
    small_font = pygame.font.SysFont("Segoe UI", 18) or pygame.font.Font(None, 18)

    board = chess.Board()

    selected_sq = None
    legal_moves_from_selected = []   # list[chess.Move]
    legal_to_sqs = set()
    capture_to_sqs = set()
    last_move = None
    status_msg = "Select a piece to see legal moves."

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
        # only allow selecting side to move
        if piece.color != board.turn:
            return
        for mv in board.legal_moves:
            if mv.from_square == selected:
                legal_moves_from_selected.append(mv)
                legal_to_sqs.add(mv.to_square)
                if board.is_capture(mv):
                    capture_to_sqs.add(mv.to_square)

    def try_make_move(from_sq, to_sq):
        nonlocal last_move, status_msg
        if from_sq is None or to_sq is None:
            return False

        # Find a legal move that matches from/to (and handle promotions)
        candidate_moves = [mv for mv in legal_moves_from_selected if mv.to_square == to_sq and mv.from_square == from_sq]
        if not candidate_moves:
            return False

        # If promotion is possible there may be multiple legal promotion moves.
        mv = None
        if len(candidate_moves) == 1:
            mv = candidate_moves[0]
        else:
            # Choose promotion piece (auto-queen)
            promo = choose_promotion_piece(board, candidate_moves[0])
            for cm in candidate_moves:
                if cm.promotion == promo:
                    mv = cm
                    break
            mv = mv or candidate_moves[0]
        
        board.push(mv)
        last_move = mv
        status_msg = f"Played: {mv.uci()}"
        return True

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board.reset()
                    selected_sq = None
                    legal_to_sqs.clear()
                    capture_to_sqs.clear()
                    legal_moves_from_selected = []
                    last_move = None
                    status_msg = "Game reset."
                elif event.key == pygame.K_u:
                    if board.move_stack:
                        board.pop()
                        selected_sq = None
                        legal_to_sqs.clear()
                        capture_to_sqs.clear()
                        legal_moves_from_selected = []
                        last_move = board.peek() if board.move_stack else None
                        status_msg = "Undid last move."

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                sq = xy_to_square(mx, my)

                if sq is None:
                    # clicked in panel area
                    continue

                # If no selection yet: select if it's your piece
                if selected_sq is None:
                    piece = board.piece_at(sq)
                    if piece and piece.color == board.turn:
                        selected_sq = sq
                        recompute_legal(selected_sq)
                        if legal_to_sqs:
                            status_msg = "Select a destination square."
                        else:
                            status_msg = "No legal moves for that piece."
                    else:
                        status_msg = "Select one of your pieces."
                else:
                    # Attempt move
                    moved = try_make_move(selected_sq, sq)
                    selected_sq = None
                    legal_to_sqs.clear()
                    capture_to_sqs.clear()
                    legal_moves_from_selected = []
                    if not moved:
                        # If clicked another of your pieces, select that instead
                        piece = board.piece_at(sq)
                        if piece and piece.color == board.turn:
                            selected_sq = sq
                            recompute_legal(selected_sq)
                            status_msg = "Switched selection."
                        else:
                            status_msg = "Illegal move."

        # Draw
        draw_board(screen, selected_sq, legal_to_sqs, capture_to_sqs, last_move)
        draw_pieces(screen, board, piece_font)
        draw_panel(screen, board, ui_font, small_font, status_msg)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

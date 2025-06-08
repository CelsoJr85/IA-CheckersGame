import pygame
import sys
import math
import copy
import time

# Initialize pygame
pygame.init()

# Constants
BOARD_SIZE = 600
WIDTH = BOARD_SIZE
HEIGHT = BOARD_SIZE
ROWS = 8
COLS = 8
SQUARE_SIZE = BOARD_SIZE // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
GOLD = (255, 215, 0)
PINK = (255, 192, 203)
YELLOW = (255, 255, 0)
DARK_GREY = (64, 64, 64)
LIGHT_GREY = (192, 192, 192)
HOVER_GREY = (96, 96, 96)


class Piece:
    PADDING = 15
    OUTLINE = 2

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def make_king(self):
        self.king = True

    def draw(self, screen):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(screen, GREY, (self.x, self.y), radius + self.OUTLINE)
        pygame.draw.circle(screen, self.color, (self.x, self.y), radius)

        if self.king:
            # Draw crown for king
            pygame.draw.circle(screen, GOLD, (self.x, self.y), radius - 10, 3)
            # Draw crown points
            for i in range(8):
                angle = i * math.pi / 4
                x_point = self.x + (radius - 15) * math.cos(angle)
                y_point = self.y + (radius - 15) * math.sin(angle)
                pygame.draw.circle(screen, GOLD, (int(x_point), int(y_point)), 3)

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()

    def __repr__(self):
        return str(self.color)


class Board:
    def __init__(self):
        self.board = []
        self.red_left = self.white_left = 12
        self.red_kings = self.white_kings = 0
        self.create_board()

    def draw_squares(self, screen):
        screen.fill(LIGHT_BROWN)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(screen, DARK_BROWN,
                                 (row * SQUARE_SIZE, col * SQUARE_SIZE,
                                  SQUARE_SIZE, SQUARE_SIZE))

    def evaluate(self):
        # Enhanced evaluation function for AI
        score = 0

        for row in self.board:
            for piece in row:
                if piece != 0:
                    piece_value = 1
                    if piece.king:
                        piece_value = 3

                    # Position bonus (center pieces are more valuable)
                    position_bonus = 0
                    if 2 <= piece.row <= 5 and 2 <= piece.col <= 5:
                        position_bonus = 0.5

                    # Promotion bonus
                    promotion_bonus = 0
                    if not piece.king:
                        if piece.color == WHITE and piece.row >= 6:
                            promotion_bonus = 0.3
                        elif piece.color == RED and piece.row <= 1:
                            promotion_bonus = 0.3

                    total_value = piece_value + position_bonus + promotion_bonus

                    if piece.color == WHITE:
                        score += total_value
                    else:
                        score -= total_value

        return score

    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = \
            self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        # Check if should become king
        if row == ROWS - 1 or row == 0:
            if not piece.king:
                piece.make_king()
                if piece.color == WHITE:
                    self.white_kings += 1
                else:
                    self.red_kings += 1

    def get_piece(self, row, col):
        return self.board[row][col]

    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, WHITE))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, RED))
                    else:
                        self.board[row].append(0)
                else:
                    self.board[row].append(0)

    def draw(self, screen):
        self.draw_squares(screen)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(screen)

    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece != 0:
                if piece.color == RED:
                    self.red_left -= 1
                else:
                    self.white_left -= 1

    def winner(self):
        if self.red_left <= 0:
            return WHITE
        elif self.white_left <= 0:
            return RED
        return None

    def has_valid_moves(self, color):
        """Check if a player has any valid moves available"""
        for piece in self.get_all_pieces(color):
            valid_moves = self.get_valid_moves(piece)
            if valid_moves:
                return True
        return False

    def is_game_over(self):
        """Check if game is over due to no pieces or no valid moves"""
        # Check if any player has no pieces left
        if self.red_left <= 0:
            return WHITE, "victory_no_pieces"
        elif self.white_left <= 0:
            return RED, "victory_no_pieces"

        # Check if any player has no valid moves (blocked)
        red_has_moves = self.has_valid_moves(RED)
        white_has_moves = self.has_valid_moves(WHITE)

        if not red_has_moves and not white_has_moves:
            # Both players blocked - draw (shouldn't happen normally)
            return None, "draw_no_moves"
        elif not red_has_moves:
            # Red player is blocked
            return WHITE, "victory_blocked"
        elif not white_has_moves:
            # White player is blocked
            return RED, "victory_blocked"

        return None, "continue"

    def get_valid_moves(self, piece):
        """Get all valid moves for a piece"""
        moves = {}

        if piece.king:
            # Kings can move in all directions and full distance
            moves.update(self._get_king_moves(piece))
        else:
            # Regular pieces can capture in any diagonal direction
            moves.update(self._get_regular_piece_moves(piece))

        return moves

    def _get_king_moves(self, piece):
        """Get all valid moves for a king piece - can move full diagonal lines"""
        moves = {}
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All diagonal directions

        for direction in directions:
            row_dir, col_dir = direction

            # Get all moves in this direction (both regular moves and captures)
            direction_moves = self._explore_direction_king(
                piece.row, piece.col, row_dir, col_dir, piece.color, []
            )
            moves.update(direction_moves)

        return moves

    def _explore_direction_king(self, start_row, start_col, row_dir, col_dir, color, captured_pieces):
        """Explore a direction for king moves with capture sequences"""
        moves = {}
        current_row = start_row + row_dir
        current_col = start_col + col_dir
        found_enemy = False
        enemy_piece = None

        while 0 <= current_row < ROWS and 0 <= current_col < COLS:
            current_piece = self.board[current_row][current_col]

            if current_piece == 0:
                # Empty square
                if found_enemy:
                    # This is a landing square after a capture
                    new_captured = captured_pieces + [enemy_piece]
                    moves[(current_row, current_col)] = new_captured

                    # Look for additional captures from this position
                    additional_moves = self._find_additional_king_captures(
                        current_row, current_col, color, new_captured
                    )

                    # Merge additional moves, keeping longer capture sequences
                    for move_pos, move_captured in additional_moves.items():
                        if move_pos not in moves or len(move_captured) > len(moves[move_pos]):
                            moves[move_pos] = move_captured

                elif not captured_pieces:
                    # Regular move without capture (only if no captures in progress)
                    moves[(current_row, current_col)] = []

            elif current_piece.color == color:
                # Own piece - can't move here, stop
                break

            else:
                # Enemy piece
                if found_enemy:
                    # Second enemy piece without landing space - can't capture
                    break
                elif current_piece not in captured_pieces:
                    # First enemy piece in this direction that hasn't been captured
                    found_enemy = True
                    enemy_piece = current_piece
                else:
                    # This piece was already captured in this sequence
                    break

            current_row += row_dir
            current_col += col_dir

        return moves

    def _find_additional_king_captures(self, start_row, start_col, color, already_captured):
        """Find additional captures for kings from a given position"""
        moves = {}
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for direction in directions:
            row_dir, col_dir = direction
            direction_moves = self._explore_direction_king(
                start_row, start_col, row_dir, col_dir, color, already_captured
            )

            # Only keep moves that capture more pieces
            for move_pos, move_captured in direction_moves.items():
                if len(move_captured) > len(already_captured):
                    if move_pos not in moves or len(move_captured) > len(moves[move_pos]):
                        moves[move_pos] = move_captured

        return moves

    def _get_regular_piece_moves(self, piece):
        """Get all valid moves for regular pieces - can capture in any diagonal direction"""
        moves = {}
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # All diagonal directions

        for direction in directions:
            row_dir, col_dir = direction
            target_row = piece.row + row_dir
            target_col = piece.col + col_dir

            # Check if target position is within bounds
            if 0 <= target_row < ROWS and 0 <= target_col < COLS:
                target_piece = self.board[target_row][target_col]

                if target_piece == 0:
                    # Empty square - regular move (only if piece can move in this direction)
                    if self._can_move_direction(piece, direction):
                        moves[(target_row, target_col)] = []

                elif target_piece.color != piece.color:
                    # Enemy piece - check if we can capture it
                    jump_row = target_row + row_dir
                    jump_col = target_col + col_dir

                    if (0 <= jump_row < ROWS and 0 <= jump_col < COLS and
                            self.board[jump_row][jump_col] == 0):
                        # Can capture this piece
                        moves[(jump_row, jump_col)] = [target_piece]

                        # Look for additional captures from landing position
                        additional_moves = self._find_additional_regular_captures(
                            jump_row, jump_col, piece.color, [target_piece]
                        )
                        moves.update(additional_moves)

        return moves

    def _can_move_direction(self, piece, direction):
        """Check if a regular piece can move in a given direction"""
        if piece.king:
            return True  # Kings can move in any direction

        row_dir, col_dir = direction

        # Regular pieces movement rules:
        # Red pieces (bottom) can move up (negative row direction)
        # White pieces (top) can move down (positive row direction)
        if piece.color == RED:
            return row_dir < 0  # Can move up
        else:  # WHITE
            return row_dir > 0  # Can move down

    def _find_additional_regular_captures(self, start_row, start_col, color, already_captured):
        """Find additional captures for regular pieces"""
        moves = {}
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Can capture in any direction

        for direction in directions:
            row_dir, col_dir = direction
            target_row = start_row + row_dir
            target_col = start_col + col_dir

            # Check bounds
            if 0 <= target_row < ROWS and 0 <= target_col < COLS:
                target_piece = self.board[target_row][target_col]

                if (target_piece != 0 and target_piece.color != color and
                        target_piece not in already_captured):
                    # Found enemy piece not yet captured
                    jump_row = target_row + row_dir
                    jump_col = target_col + col_dir

                    if (0 <= jump_row < ROWS and 0 <= jump_col < COLS and
                            self.board[jump_row][jump_col] == 0):
                        # Can capture this piece
                        new_captured = already_captured + [target_piece]
                        moves[(jump_row, jump_col)] = new_captured

                        # Recursively look for more captures
                        if len(new_captured) < 8:  # Prevent infinite recursion
                            additional_moves = self._find_additional_regular_captures(
                                jump_row, jump_col, color, new_captured
                            )
                            # Only keep moves with more captures
                            for move_pos, move_captured in additional_moves.items():
                                if len(move_captured) > len(new_captured):
                                    if move_pos not in moves or len(move_captured) > len(moves[move_pos]):
                                        moves[move_pos] = move_captured

        return moves


class AI:
    def __init__(self, depth=4):
        self.depth = depth

    def get_all_moves(self, board, color):
        moves = []
        for piece in board.get_all_pieces(color):
            valid_moves = board.get_valid_moves(piece)
            for move, capture in valid_moves.items():
                moves.append((piece, move, capture))

        # Prioritize capture moves (mandatory captures)
        capture_moves = [move for move in moves if len(move[2]) > 0]
        if capture_moves:
            return capture_moves
        return moves

    def simulate_move(self, board, piece, move, captured):
        # Create deep copy of board
        temp_board = copy.deepcopy(board)
        temp_piece = temp_board.get_piece(piece.row, piece.col)
        temp_board.move(temp_piece, move[0], move[1])

        if captured:
            # Remove captured pieces
            pieces_to_remove = []
            for captured_piece in captured:
                pieces_to_remove.append(temp_board.get_piece(captured_piece.row, captured_piece.col))
            temp_board.remove(pieces_to_remove)

        return temp_board

    def minimax(self, board, depth, maximizing, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or board.winner() is not None:
            return board.evaluate(), None

        best_move = None

        if maximizing:
            max_eval = float('-inf')
            color = WHITE

            moves = self.get_all_moves(board, color)
            moves.sort(key=lambda x: len(x[2]), reverse=True)  # Prioritize captures

            for piece, move, captured in moves:
                new_board = self.simulate_move(board, piece, move, captured)
                evaluation, _ = self.minimax(new_board, depth - 1, False, alpha, beta)

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = (piece, move, captured)

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            return max_eval, best_move

        else:
            min_eval = float('inf')
            color = RED

            moves = self.get_all_moves(board, color)
            moves.sort(key=lambda x: len(x[2]), reverse=True)  # Prioritize captures

            for piece, move, captured in moves:
                new_board = self.simulate_move(board, piece, move, captured)
                evaluation, _ = self.minimax(new_board, depth - 1, True, alpha, beta)

                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = (piece, move, captured)

                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def get_best_move(self, board):
        _, best_move = self.minimax(board, self.depth, True)
        return best_move


class Game:
    def __init__(self, screen):
        self._init()
        self.screen = screen
        self.ai = AI(depth=4)
        self.ai_mode = True  # True = play against AI, False = two players
        self.thinking = False

    def update(self):
        self.board.draw(self.screen)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def _init(self):
        self.selected = None
        self.board = Board()
        self.turn = RED  # Human player starts
        self.valid_moves = {}
        self.thinking = False

    def winner(self):
        winner_info = self.board.is_game_over()
        if winner_info[1] != "continue":
            return winner_info
        return None, "continue"

    def reset(self):
        self._init()

    def toggle_mode(self):
        self.ai_mode = not self.ai_mode
        self.reset()

    def select(self, row, col):
        if self.thinking:
            return False

        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)

        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)

            # Filter moves to prioritize captures (mandatory capture rule)
            capture_moves = {k: v for k, v in self.valid_moves.items() if len(v) > 0}
            if capture_moves:
                self.valid_moves = capture_moves

            return True

        return False

    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
        else:
            return False

        return True

    def ai_move(self):
        if self.ai_mode and self.turn == WHITE and not self.thinking:
            # Check if AI has any valid moves
            if not self.board.has_valid_moves(WHITE):
                return  # AI can't move, game will end

            self.thinking = True
            pygame.display.update()

            # Small delay to show AI is "thinking"
            pygame.time.wait(500)

            best_move = self.ai.get_best_move(self.board)

            if best_move:
                piece, move, captured = best_move
                self.board.move(piece, move[0], move[1])
                if captured:
                    self.board.remove(captured)
                self.change_turn()

            self.thinking = False

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            # Different colors for different types of moves
            if (row, col) in self.valid_moves:
                captures = self.valid_moves[(row, col)]
                if len(captures) > 0:
                    # Red circle for capture moves
                    pygame.draw.circle(self.screen, RED,
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                        row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)
                    # Add number indicating how many pieces will be captured
                    font = pygame.font.Font(None, 24)
                    text = font.render(str(len(captures)), True, WHITE)
                    text_rect = text.get_rect(center=(col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                      row * SQUARE_SIZE + SQUARE_SIZE // 2))
                    self.screen.blit(text, text_rect)
                else:
                    # Blue circle for regular moves
                    pygame.draw.circle(self.screen, BLUE,
                                       (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                        row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)

    def change_turn(self):
        self.valid_moves = {}
        self.selected = None
        if self.turn == RED:
            self.turn = WHITE
        else:
            self.turn = RED


def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Checkers Game - Advanced Rules')
    clock = pygame.time.Clock()
    game = Game(screen)
    game_over = False
    winner_info = (None, "continue")

    running = True
    while running:
        clock.tick(60)

        # Check for game over conditions only if game is not already over
        if not game_over:
            winner_info = game.winner()
            if winner_info[1] != "continue":
                game_over = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if not game.thinking and not game_over:
                    pos = pygame.mouse.get_pos()
                    row, col = get_row_col_from_mouse(pos)
                    # Only allow clicks if it's human player's turn
                    if not game.ai_mode or game.turn == RED:
                        game.select(row, col)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press R to restart
                    game.reset()
                    game_over = False
                    winner_info = (None, "continue")
                if event.key == pygame.K_m:  # Press M to toggle mode
                    game.toggle_mode()
                    game_over = False
                    winner_info = (None, "continue")

        # Execute AI move if it's AI's turn and game is not over
        if (game.ai_mode and game.turn == WHITE and not game.thinking and not game_over):
            game.ai_move()

        # Always update the game display
        game.update()

        # Show game over screen if game is over
        if game_over:
            winner_color, reason = winner_info

            # Determine winner message based on color and reason
            font = pygame.font.Font(None, 72)
            medium_font = pygame.font.Font(None, 48)
            small_font = pygame.font.Font(None, 36)

            if reason == "victory_no_pieces":
                if winner_color == RED:
                    if game.ai_mode:
                        main_text = "YOU WON!"
                        sub_text = "All AI pieces captured!"
                        text_color = GREEN
                    else:
                        main_text = "RED WINS!"
                        sub_text = "All white pieces captured!"
                        text_color = RED
                else:  # WHITE wins
                    if game.ai_mode:
                        main_text = "AI WON!"
                        sub_text = "All your pieces captured!"
                        text_color = BLUE
                    else:
                        main_text = "WHITE WINS!"
                        sub_text = "All red pieces captured!"
                        text_color = BLACK

            elif reason == "victory_blocked":
                if winner_color == RED:
                    if game.ai_mode:
                        main_text = "YOU WON!"
                        sub_text = "AI has no valid moves!"
                        text_color = GREEN
                    else:
                        main_text = "RED WINS!"
                        sub_text = "White pieces blocked!"
                        text_color = RED
                else:  # WHITE wins
                    if game.ai_mode:
                        main_text = "AI WON!"
                        sub_text = "You have no valid moves!"
                        text_color = BLUE
                    else:
                        main_text = "WHITE WINS!"
                        sub_text = "Red pieces blocked!"
                        text_color = BLACK

            elif reason == "draw_no_moves":
                main_text = "DRAW!"
                sub_text = "No valid moves for either player!"
                text_color = GREY

            # Semi-transparent background
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            screen.blit(overlay, (0, 0))

            # Render and position texts
            main_surface = font.render(main_text, True, text_color)
            sub_surface = medium_font.render(sub_text, True, WHITE)
            continue_surface = small_font.render("Press R to start a new game", True, WHITE)
            mode_surface = small_font.render("Press M to change game mode", True, GREY)

            # Center the texts
            main_rect = main_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
            sub_rect = sub_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 10))
            continue_rect = continue_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
            mode_rect = mode_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 80))

            screen.blit(main_surface, main_rect)
            screen.blit(sub_surface, sub_rect)
            screen.blit(continue_surface, continue_rect)
            screen.blit(mode_surface, mode_rect)

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
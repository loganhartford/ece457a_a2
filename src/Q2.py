import numpy as np
import random

def create_board():
    return np.array([
        [10, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, -10],
        ])

def location_valid(board, row, col, player):
    return (0 <= row < 4 and 0 <= col < 4) and (board[row, col] * player >= 0)

def generate_moves(board, player):
    moves = []
    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (-1, -1), # up-left
        (-1, 1),  # up-right
        (1, -1),  # down-left
        (1, 1)    # down-right
    ]

    # Iterate through the board to find the player's stones
    for row in range(4):
        for col in range(4):
            # If stone belongs to player
            if board[row, col] * player > 0:
                for direction in directions:
                    new_row, new_col = row + direction[0], col + direction[1]

                    # Check if the move is within the board boundaries
                    if location_valid(board, new_row, new_col, player):
                            moves.append(((row, col), direction))
    
    return moves

def apply_move(board, move, player):
    new_board = np.copy(board)
    (start_row, start_col), direction = move
    num_stones = abs(new_board[start_row, start_col])

    # Clear the starting position
    new_board[start_row, start_col] = 0

    # Distribute the stones along the specified direction
    current_row, current_col = start_row, start_col
    for step in range(1, num_stones + 1):
        if not num_stones:
            break
        current_row += direction[0]
        current_col += direction[1]

        if location_valid(new_board, current_row, current_col, player):
            place = min(num_stones, step)
            new_board[current_row, current_col] += place * player
            num_stones -= place
        else:
            current_row -= direction[0]
            current_col -= direction[1]
            new_board[current_row, current_col] += num_stones * player
            break

    return new_board

def minimax(board, depth, alpha, beta, maximizingPlayer, eval_function):
    # Base case: if the maximum depth is reached or the game is over
    if depth == 0 or len(generate_moves(board, 1)) == 0 or len(generate_moves(board, -1)) == 0:
        return eval_function(board)

    if maximizingPlayer:
        maxEval = float('-inf')
        for move in generate_moves(board, 1):
            new_board = apply_move(board, move, 1)
            
            # Recursively call minimax
            eval = minimax(new_board, depth - 1, alpha, beta, False, eval_function)
            
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            
            # Alpha-Beta pruning
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for move in generate_moves(board, -1):
            new_board = apply_move(board, move, -1)
            
            # Recursively call minimax
            eval = minimax(new_board, depth - 1, alpha, beta, True, eval_function)

            minEval = min(minEval, eval)
            beta = min(beta, eval)

            # Alpha-Beta pruning
            if beta <= alpha:
                break
        return minEval


# Evaluation Funcitons
def eval_move_delta(board):
    player1_moves = len(generate_moves(board, player=1))
    player2_moves = len(generate_moves(board, player=-1))
    
    return player1_moves - player2_moves

def eval_opponent_moves(board, player):
    opponent = -player
    opponent_moves = len(generate_moves(board, opponent))
    return -opponent_moves

def eval_position_control(board, player):
    player_positions = np.sum(board * player > 0)
    opponent_positions = np.sum(board * -player > 0)
    return player_positions - opponent_positions

def eval_concentration(board, player):
    player_max_concentration = np.max(board * (board * player > 0))
    opponent_max_concentration = np.max(board * (board * -player > 0))
    return player_max_concentration - opponent_max_concentration

def eval_available_captures(board, player):
    moves = generate_moves(board, player)
    capture_opportunities = sum(1 for move in moves if board[move[0][0], move[0][1]] * player > 0)
    return capture_opportunities

def eval_stones_with_moves(board, player):
    moves = generate_moves(board, player)
    stones_with_moves = len({(move[0][0], move[0][1]) for move in moves})
    return stones_with_moves


# Game
def play_game(board, depth, eval_function, player1_agent, player2_agent, verbose=False, limit_epochs=True):
    current_player = 1  # Player 1 starts
    game_over = False
    turns = 0

    while not game_over:
        turns += 1
        if (limit_epochs) and (turns % 100 == 0):
            break

        if verbose:
            print("Current board state:")
            print(board)

        if current_player == 1:
            # Player 1's turn
            if verbose: print("Player 1's turn")
            moves = generate_moves(board, current_player)
            if not moves:
                print("Player 1 has no legal moves. Player 2 wins!")
                break

            # Use the specified strategy for Player 1
            board = player1_agent(board, current_player, depth, eval_function)
        else:
            # Player 2's turn
            if verbose: print("Player 2's turn")
            moves = generate_moves(board, current_player)
            if not moves:
                print("Player 2 has no legal moves. Player 1 wins!")
                break

            # Use the specified strategy for Player 2
            board = player2_agent(board, current_player, depth, eval_function)

        # Check if the game is over (no moves left for either player)
        if not generate_moves(board, 1) and not generate_moves(board, -1):
            print("No legal moves left for either player. The game is a draw.")
            break
        
        # Switch players
        current_player = -current_player

    return board, turns

# Agents
def minimax_agent(board, player, depth, eval_function):
    moves = generate_moves(board, player)
    best_move = None
    best_value = float('-inf') if player == 1 else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in moves:
        new_board = apply_move(board, move, player)
        move_value = minimax(new_board, depth, alpha, beta, player == -1, eval_function)
        
        if (player == 1 and move_value > best_value) or (player == -1 and move_value < best_value):
            best_value = move_value
            best_move = move

    return apply_move(board, best_move, player)


def first_move_agent(board, player, depth, eval_function):
    moves = generate_moves(board, player)
    return apply_move(board, moves[0], player)

def do_nothing_agent(board, player, depth, eval_function):
    return board

def random_agent(board, player, *args):
    moves = generate_moves(board, player)
    random_move = random.choice(moves)
    return apply_move(board, random_move, player)
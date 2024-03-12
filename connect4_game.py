from tictactoe import Board
import numpy as np
import pickle
import math
import random

# Place the counter at the bottom of the selected column
def get_placement(board, index):
    column = board.board[index]
    max_index = -1
    for i in range(len(column)):
        if column[i] == 0:
            max_index = i
    return [index, max_index]

# Get all available moves
def get_available_moves(board):
    zero_indices = []
    for i in range(len(board.board)):
        if 0 in board.board[i]:
            zero_indices.append(i)
    return zero_indices

class random_player:
    def __init__(self):
        self.name = "Random Player"
        
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = random.choice(available_moves)
        return get_placement(board, move)

# Human contolled player
class human_player:
    def __init__(self):
        self.name = "Sean"
    
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = None

        # Get user input
        while move not in available_moves or move == None:
            try:
                move_input = input("Enter a column to place a counter:")
                move = int(move_input)
                if move in available_moves:
                    return get_placement(board, move)
                else: print("Invalid Move.")
            except Exception as e:
                print("Invalid Move.")

# Flip the x and y axes of the board
def transpose_board(board):
    transposed_board = [[0 for _ in range(len(board.board))] for _ in range(len(board.board[0]))]
    for i in range(len(board.board)):
        for j in range(len(board.board[0])):
            transposed_board[j][i] = board.board[i][j]
    return transposed_board

def compute_value(unbroken_pieces, unbroken_spaces):
    if unbroken_pieces >= 3:
        if unbroken_spaces >= 1:
            return .9
        else: return 0
    if unbroken_pieces >= 2:
        if unbroken_spaces >= 2:
            return .5
        else: return 0
    else: return 0

# Get heuristics for vertical connections   
def check_verticals(board, player1=True):
    available_moves = get_available_moves(board)
    x_scores = []
    y_scores = []
    
    for i in range(len(board.board)):
        unbroken_x_pieces = 0
        unbroken_x_spaces = 0
        unbroken_y_pieces = 0
        unbroken_y_spaces = 0
        
        for index, space in enumerate(board.board[i]):
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves:
                    if get_placement(board, index)[1] == i:
                        unbroken_x_spaces = unbroken_x_spaces + 1
                        unbroken_y_spaces = unbroken_y_spaces + 1
                else: 
                    x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                    y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                    unbroken_x_pieces = 0
                    unbroken_x_spaces = 0
                    unbroken_y_pieces = 0
                    unbroken_y_spaces = 0
            if space == 2:
                unbroken_y_pieces = unbroken_y_pieces + 1
                x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                unbroken_x_pieces = 0
                unbroken_x_spaces = 0
        x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
        y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
    
    # Get total score
    if x_scores.count(0.9) > 1:
        return math.inf
    elif y_scores.count(0.9) > 1:
        return -math.inf
    return np.sum(x_scores) - np.sum(y_scores)

# Check for three in a row with open spaces to either side
def double_open_three(board, col, row, available_moves):
    pattern = [0, 1, 1, 1, 0]
    for i in range(len(col) - len(pattern) + 1):
        if tuple(col[i:i+len(pattern)]) == tuple(pattern):
            if i in available_moves:
                if get_placement(board, i)[1] == row and get_placement(board, i+len(pattern)-1)[1] == row:
                    return True
    return False

# Get heuristics for horizontal connections
def check_horizontals(board):
    transposed_board = transpose_board(board)
    available_moves = get_available_moves(board)
    x_scores = []
    y_scores = []
    
    for i in range(len(transposed_board)):
        unbroken_x_pieces = 0
        unbroken_x_spaces = 0
        unbroken_y_pieces = 0
        unbroken_y_spaces = 0

        # Calculate the value of each row
        if double_open_three(board, transposed_board[i], i, available_moves):
            return math.inf
        for index, space in enumerate(transposed_board[i]):
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves:
                    if get_placement(board, index)[1] == i:
                        unbroken_x_spaces = unbroken_x_spaces + 1
                        unbroken_y_spaces = unbroken_y_spaces + 1
                else: 
                    x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                    y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                    unbroken_x_pieces = 0
                    unbroken_x_spaces = 0
                    unbroken_y_pieces = 0
                    unbroken_y_spaces = 0
            if space == 2:
                unbroken_y_pieces = unbroken_y_pieces + 1
                print()
                x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                unbroken_x_pieces = 0
                unbroken_x_spaces = 0
        x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
        y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
    
    # Get total score
    if x_scores.count(0.9) > 1:
        return math.inf
    elif y_scores.count(0.9) > 1:
        return -math.inf
    print(np.sum(x_scores), np.sum(y_scores))
    return np.sum(x_scores) - np.sum(y_scores)

# Get heuristics for diagonal connections
def check_diagonals(board, player1=True):
    rows = len(board.board)
    cols = len(board.board[0])
    available_moves = get_available_moves(board)
    x_scores = []
    y_scores = []

    # Check first diagonal direction 
    for i in range(rows + cols - 1):
        unbroken_x_pieces = 0
        unbroken_x_spaces = 0
        unbroken_y_pieces = 0
        unbroken_y_spaces = 0
        
        for index, j in enumerate(range(max(0, i - cols + 1), min(rows, i + 1))):
            space = board.board[j][cols - i + j - 1]
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves:
                    if get_placement(board, index)[1] == i:
                        unbroken_x_spaces = unbroken_x_spaces + 1
                        unbroken_y_spaces = unbroken_y_spaces + 1
                else: 
                    x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                    y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                    unbroken_x_pieces = 0
                    unbroken_x_spaces = 0
                    unbroken_y_pieces = 0
                    unbroken_y_spaces = 0
            if space == 2:
                unbroken_y_pieces = unbroken_y_pieces + 1
                x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
                unbroken_x_pieces = 0
                unbroken_x_spaces = 0
        x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_spaces))
        y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
        
    # Check second diagonal direction 
    for i in range(rows + cols - 1):
        unbroken_x_pieces = 0
        unbroken_x_spaces = 0
        unbroken_y_pieces = 0
        unbroken_y_spaces = 0
        
        for index, j in enumerate(range(max(0, i - cols + 1), min(rows, i + 1))):
            space = board.board[j][i - j]
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_pieces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves:
                    if get_placement(board, index)[1] == i:
                        unbroken_x_spaces = unbroken_x_spaces + 1
                        unbroken_y_spaces = unbroken_y_spaces + 1
                else: 
                    x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_pieces))
                    y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_pieces))
                    unbroken_x_pieces = 0
                    unbroken_x_spaces = 0
                    unbroken_y_pieces = 0
                    unbroken_y_spaces = 0
            if space == 2:
                x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_pieces))
                unbroken_x_pieces = 0
                unbroken_x_spaces = 0
        x_scores.append(compute_value(unbroken_x_pieces, unbroken_x_pieces))
        y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_pieces))
    
    # Get total score
    if x_scores.count(0.9) > 1:
        return math.inf
    elif y_scores.count(0.9) > 1:
        return -math.inf
    return np.sum(x_scores) - np.sum(y_scores)
            
# Player using minimax
class minimax_player:
    def __init__(self):
        self.name = "Minimax"
        
    #def get_state_heuristic(board):
        
        
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first move
        if depth == 0:
            return 3, 0
        
        # Check if the game is over
        try:
            result = board.result()
            if result == 1:
                return None, 1
            elif result == 2:
                return None, -1
        except Exception as e:
            if str(e) == "Both X and O have 3 pieces in a row.":
                return None, 0
            else: print(f"A minimax Error Occured: {e}")
        
        # If max depth is reached, check heuristics
        #if depth == 5:
            
                
        # Calculate move for maximiser
        if is_maximising:
            max_value = -math.inf
            best_move = None
            
            # get value of each move
            available_moves = get_available_moves(board)
            for move in available_moves:
                new_board = board.copy()
                new_board.push(get_placement(board, move))
                best_move, value = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value > max_value:
                    max_value = value
                    best_move = move
                
                # Prune search
                if max_value > alpha:
                    alpha = max_value
                if alpha >= beta:
                    break
                    
            return best_move, max_value
        
        # Calculate move for minimiser
        else:
            min_value = math.inf
            best_move = None
            
            # Get value of each move
            for move in available_moves:
                new_board = board.copy()
                new_board.push(get_placement(board, move))
                best_move, value = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value < min_value:
                    min_value = value
                    best_move = move
                
                # Prune search
                if min_value < beta:
                    beta = min_value
                if alpha >= beta:
                    break
                
            return best_move, min_value
         
    # Return the next move for the player
    def get_next_move(self, board):
        move = self.minimax(board, -math.inf, math.inf, True, 0)
        return move
                
def play_tictactoe(board, player1, player2):
    print(f"Game Starting. \nPlayers: {player1.name}, {player2.name}\n")
    
    # While the game is not over let each player move
    while board.result() == None:
        player1_move = player1.get_next_move(board)
        board.push(player1_move)
        if board.result() != None:
            print(board, "\n")
            break
        player2_move = player2.get_next_move(board)
        board.push(player2_move)
        print(board, "\n")
        
    # Print the winner
    if board.result() == 1:
        print(f"Winner = {player1.name}")
    elif board.result() == 2: print(f"Winner = {player2.name}")
    else: print("Tie Game")
    
tictactoe_board = Board(dimensions=(7, 6), x_in_a_row=4)
#print(get_available_moves(tictactoe_board))
#player1 = human_player()
#playa2 = random_player()
#playa2 = Qlearning_player(policy_name='Q_learning_agent')
#player1.train_Qlearning_agent(10000)
tictactoe_board.push(get_placement(tictactoe_board, 0))
tictactoe_board.push(get_placement(tictactoe_board, 3))
tictactoe_board.push(get_placement(tictactoe_board, 1))
tictactoe_board.push(get_placement(tictactoe_board, 5))
tictactoe_board.push(get_placement(tictactoe_board, 2))
tictactoe_board.push(get_placement(tictactoe_board, 5))
tictactoe_board.push(get_placement(tictactoe_board, 0))
tictactoe_board.push(get_placement(tictactoe_board, 4))
print(tictactoe_board)
print("vert score: ", check_verticals(tictactoe_board))
print("horz score: ", check_horizontals(tictactoe_board))
print("diag score: ", check_diagonals(tictactoe_board))

#play_tictactoe(tictactoe_board, player1, playa2)
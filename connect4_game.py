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

class Default_player:
    def __init__(self, optimality = 1, is_player_1 = True):
        self.name = "Default Player"
        self.optimality = optimality
        self.is_player_1 = is_player_1
        
    # Get the reward for taking an action
    def get_state_reward(self, board_state, opposing_player=False):
        is_player_1 = self.is_player_1
        if opposing_player:
            is_player_1 = is_player_1
        result = board_state.result()
        if result == 1 and is_player_1:
            return 1
        elif result == 1 and not is_player_1:
            return -1
        elif result == 2 and is_player_1:
            return -1
        elif result == 2 and not is_player_1:
            return 1
        return 0
    
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        best_move = None
        random_num = random.randint(0,int(1/self.optimality)-1)
        
        # Check for winning moves
        for move in available_moves:
            new_board = board.copy()
            new_board.push(move)
            result = self.get_state_reward(new_board)
            if result > 0 and random_num==1:
                return move
            
        # Block loosing moves
        for move in available_moves:
            new_board = board.copy()
            if self.is_player_1:
                new_board.turn = 2
            else: new_board.turn = 1
            new_board.push(move)
            result = self.get_state_reward(new_board)
            if result < 0 and random_num==1:
                return move
            else: best_move = move
        return best_move

class Random_player:
    def __init__(self):
        self.name = "Random Player"
        
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = random.choice(available_moves)
        return get_placement(board, move)

# Human contolled player
class Human_player:
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

# Get the score associated with a row of pieces
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
def check_verticals(board):
    available_moves = get_available_moves(board)
    x_scores = []
    y_scores = []
    
    for i in range(len(board.board)):
        unbroken_x_pieces = 0
        unbroken_x_spaces = 0
        unbroken_y_pieces = 0
        unbroken_y_spaces = 0
        
        for space in board.board[i]:
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves:
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
    x_pattern = [0, 1, 1, 1, 0]
    y_pattern = [0, 2, 2, 2, 0]
    for i in range(len(col) - len(x_pattern) + 1):
        if tuple(col[i:i+len(x_pattern)]) == tuple(x_pattern):
            if i in available_moves:
                if get_placement(board, i)[1] == row and get_placement(board, i+len(x_pattern)-1)[1] == row:
                    return math.inf
        elif tuple(col[i:i+len(y_pattern)]) == tuple(y_pattern):
            if i in available_moves:
                if get_placement(board, i)[1] == row and get_placement(board, i+len(y_pattern)-1)[1] == row:
                    return -math.inf
    return 0

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
            return double_open_three(board, transposed_board[i], i, available_moves)
        for index, space in enumerate(transposed_board[i]):
            if space == 1:
                unbroken_x_pieces = unbroken_x_pieces + 1
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves and (i+1 >= len(transposed_board) or transposed_board[i+1][index] != 0):
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

# Get heuristics for diagonal connections
def check_diagonals(board):
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
                if i in available_moves and ((cols - i + j) >= len(board.board[j]) or board.board[j][cols - i + j] != 0):
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
                y_scores.append(compute_value(unbroken_y_pieces, unbroken_y_spaces))
                unbroken_y_pieces = 0
                unbroken_y_spaces = 0
            if space == 0:
                if i in available_moves and ((i - j + 1) >= len(board.board[j]) or board.board[j][i - j + 1] != 0):
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

def get_state_heuristic(board):
    horizontal_score = check_horizontals(board)
    vertical_score = check_verticals(board)
    diagonal_score = check_diagonals(board)
    total_score = np.sum([horizontal_score, vertical_score, diagonal_score])
    return total_score
            
# Player using minimax
class Minimax_player:
    def __init__(self):
        self.name = "Minimax"
        self.played_moves = 0
        
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first move
        if self.played_moves == 0:
            return 3, 0
        
        # Check if the game is over
        try:
            result = board.result()
            if result == 1:
                return None, math.inf
            elif result == 2:
                return None, -math.inf
        except Exception as e:
            if str(e) == "Both X and O have 3 pieces in a row.":
                return None, 0
            else: print(f"A minimax Error Occured: {e}")
        
        # If max depth is reached, check heuristics
        if depth == 6:
            return None, get_state_heuristic(board)
                
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
            available_moves = get_available_moves(board)
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
        self.played_moves = self.played_moves+1
        return get_placement(board, move[0])
    
class Q_learning_player:
    def __init__(self, name="Connect_Four_Q_learning_agent", policy_name=None, alpha=.2, gamma=.9, is_player_1=True):
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.policy_name = policy_name
        self.is_player_1 = is_player_1
        self.prev_states = []
        if policy_name != None:
            self.policy = self.load_policy(policy_name)
        else: self.policy = {}
        self.played_moves = 0
    
    # Save Qtable
    def save_policy(self, policy_name):
        filehandler = open(policy_name,"wb")
        pickle.dump(self.policy, filehandler)
        filehandler.close()
    
    # Load a Qtable
    def load_policy(self, policy_name):
        file = open(policy_name,'rb')
        policy = pickle.load(file)
        file.close()
        return policy
    
    # Print a progress bar during training
    def print_progress_bar(self, iteration, iterations, bar_length=50):
        progress = iteration/iterations
        arrow = '-' * int(progress * bar_length - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\rTraining Q-learning Agent: [{arrow + spaces}] {progress:.2%}', end='', flush=True)
    
    # Get the reward for taking an action
    def get_state_reward(self, board_state):
        is_player_1 = self.is_player_1
        try:
            result = board_state.result()
            if result == 1 and is_player_1:
                return 1
            elif result == 1 and not is_player_1:
                return -1
            elif result == 2 and is_player_1:
                return -1
            elif result == 2 and not is_player_1:
                return 1
        except Exception as e:
            if str(e) == "Both X and O have 3 pieces in a row.":
                return 0
            else: print(f"A Q Learning state reward Error Occured: {e}")
        return None
    
    def delete_prev_states(self):
        self.prev_states = []
        
    # Update Q-values 
    def update_policy(self, reward):
        for prev_state in reversed(self.prev_states):
            if prev_state in self.policy:
                old_value = self.policy[prev_state]
            else: old_value = 0
            new_value = (1-self.alpha)*old_value + self.alpha*(self.gamma*(reward))
            self.policy[prev_state] = new_value
            reward = self.policy[prev_state]
            
    # Train Q-learning agent
    def train_Qlearning_agent(self, iterations):
        
        # Create an agent to compete against
        agent1 = self
        agent2 = Q_learning_player(is_player_1=False)
        
        # Play a new game for each iteration
        for iteration in range(iterations):
            if (iteration % 10) == 0:
                self.print_progress_bar(iteration, iterations)
                
            board = Board(dimensions=(7, 6), x_in_a_row=4)
            available_moves = get_available_moves(board)
            while agent1.get_state_reward(board) == None and len(available_moves) > 0:
                
                # play Agent 1's move and update past states
                agent1_move = agent1.get_next_move(board)
                board.push(agent1_move)
                agent1.prev_states.append(get_board_hash(board))
                
                # Check if the game is over
                if board.result() != None:
                    update_policies(board, agent1, agent2)
                    agent1.delete_prev_states()
                    agent2.delete_prev_states()
                    break
                
                # play Agent 2's move and update past states
                available_moves = get_available_moves(board)
                agent2_move = agent2.get_next_move(board)
                board.push(agent2_move)
                agent2.prev_states.append(get_board_hash(board))
                
                # Check if the game is over
                if board.result() != None:
                    update_policies(board, agent1, agent2)
                    agent1.delete_prev_states()
                    agent2.delete_prev_states()
                    break
                
        # Merge & save the policies
        agent1.policy = merge_policies(agent1.policy, agent2.policy)
        agent1.save_policy(self.name)
    
    def get_next_move(self, board):
        value = None
        max_value = -math.inf
        Q_table = self.policy
        available_moves = get_available_moves(board)
        if self.played_moves == 0 and self.is_player_1:
            self.played_moves = self.played_moves+1
            return get_placement(board, 3)
        if self.played_moves == 0 and not self.is_player_1:
            self.played_moves = self.played_moves+1
            return get_placement(board, 4)
        if self.played_moves == 1 and self.is_player_1:
            self.played_moves = self.played_moves+1
            return get_placement(board, 2)
        if self.played_moves == 1 and not self.is_player_1:
            self.played_moves = self.played_moves+1
            return get_placement(board, 3)
        
        # Check the value of each available move
        for move in available_moves:
            new_board = board.copy()
            new_board.push(get_placement(board, move))
            
            # Check if move leads to a terminal state
            if self.get_state_reward(new_board) != None:
                value = self.get_state_reward(new_board)
                
            # Check if state is in table
            elif get_board_hash(new_board) in Q_table:
                value = Q_table[get_board_hash(new_board)]
            else:
                value = 0
                Q_table[get_board_hash(new_board)] = 0
            
            # Select the highest value move
            if value > max_value:
                max_value = value
                best_move = move
        return get_placement(board, best_move)

# Get a string representation of the board
def get_board_hash(board):
    horizontal_score = check_horizontals(board)
    vertical_score = check_verticals(board)
    diagonal_score = check_diagonals(board)
    heuristic_scores = [horizontal_score, vertical_score, diagonal_score]
    hash = ','.join(map(str, heuristic_scores))
    return hash
    
def update_policies(board, agent1, agent2):
    if board.result() == 1:
        agent1.update_policy(1)
        agent2.update_policy(-1)
    elif board.result() == 2:
        agent1.update_policy(-1)
        agent2.update_policy(1)
    else:
        agent1.update_policy(0)
        agent2.update_policy(0)

def merge_policies(policy1, policy2):
    merged_policy = policy1.copy()
    merged_policy.update(policy2)
    return merged_policy
                
def play_connect_four(board, player1, player2):
    print(f"Game Starting. \nPlayers: {player1.name}, {player2.name}\n")
    
    # While the game is not over let each player move
    while board.result() == None:
        player1_move = player1.get_next_move(board)
        #print(player1_move)
        board.push(player1_move)
        print(f"\n{board}\n")
        if board.result() != None:
            break
        player2_move = player2.get_next_move(board)
        board.push(player2_move)
        print(f"\n{board}\n")
        print("Board Heuristic:", get_state_heuristic(board))
        
    # Print the winner
    if board.result() == 1:
        print(f"Winner = {player1.name}")
    elif board.result() == 2: print(f"Winner = {player2.name}")
    else: print("Tie Game")
    
tictactoe_board = Board(dimensions=(7, 6), x_in_a_row=4)
#print(get_available_moves(tictactoe_board))
playa2 = Default_player(is_player_1=False)
#playa1 = Human_player()
#playa1 = Random_player()
playa1 = Minimax_player()
#playa1 = Q_learning_player(policy_name="Connect_Four_Q_learning_agent")
#playa2 = Q_learning_player(policy_name="Connect_Four_Q_learning_agent", is_player_1=False)
#playa1.train_Qlearning_agent(500)
#print(playa2.policy)

play_connect_four(tictactoe_board, playa1, playa2)
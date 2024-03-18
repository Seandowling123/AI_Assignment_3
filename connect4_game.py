from tictactoe import Board
import numpy as np
import pickle
import math
import random
import csv

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
        self.played_moves = 0
        
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
        optimal = random.random() < self.optimality
        
        # Check for winning moves
        for move in available_moves:
            new_board = board.copy()
            new_board.push(get_placement(new_board, move))
            result = self.get_state_reward(new_board)
            if result > 0 and optimal:
                return get_placement(board, move)
            
        # Block loosing moves
        for move in available_moves:
            new_board = board.copy()
            if self.is_player_1:
                new_board.turn = 2
            else: new_board.turn = 1
            new_board.push(get_placement(new_board, move))
            result = self.get_state_reward(new_board)
            if result < 0 and optimal:
                return get_placement(board, move)
        return get_placement(board, random.choice(available_moves))

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
    def __init__(self, is_player_1=True, depth=4):
        self.name = "Minimax"
        self.is_player_1 = is_player_1
        self.played_moves = 0
        self.depth = depth
        
    # Get the reward for taking an action
    def get_state_reward(self, board_state):
        is_player_1 = self.is_player_1
        result = board_state.result()
        if result == 1 and is_player_1:
            return None, math.inf
        elif result == 1 and not is_player_1:
            return None, -math.inf
        elif result == 2 and is_player_1:
            return None, -math.inf
        elif result == 2 and not is_player_1:
            return None, math.inf
        return None, 0
        
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first moves
        if self.played_moves == 0:
            return 3, 0
        if self.played_moves == 1:
            return 4, 0
        if  self.played_moves == 2 and self.is_player_1:
            return 5, 0
        
        # Check if the game is over
        if board.result() != None:
            return self.get_state_reward(board)
        
        # If max depth is reached, check heuristics
        if depth == self.depth:
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
    def __init__(self, name="Connect_Four_Q_learning_agent", policy_name=None, alpha=.2, gamma=.9, epsilon=.3, training=False, is_player_1=True):
        self.name = name
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = .00001
        self.final_epsilon = .0001
        self.policy_name = policy_name
        self.is_player_1 = is_player_1
        self.training = training
        self.prev_states = []
        self.prev_actions = []
        if policy_name != None:
            self.policy = self.load_policy(policy_name)
        else: self.policy = {}
        self.played_moves = 0
    
    # Save Qtable
    def save_policy(self, policy_name):
        filehandler = open("Connect_four_Q_learning_agents/"+policy_name, 'wb')
        pickle.dump(self.policy, filehandler)
        filehandler.close()
    
    # Load a Qtable
    def load_policy(self, policy_name):
        file = open(policy_name, 'rb')
        policy = pickle.load(file)
        file.close()
        return policy
    
    # Print a progress bar during training
    def print_progress_bar(self, iteration, iterations, bar_length=50):
        progress = iteration/iterations
        arrow = '-' * int(progress * bar_length - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        print(f'\rTraining Q-learning Agent: [{arrow + spaces}] {progress:.2%}', end='', flush=True)
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * (1 / (1 + self.decay_rate)))
    
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
        self.prev_actions = []
        
    # Update Q-table 
    def update_policy(self, reward):
        prev_states = list(reversed(self.prev_states))
        prev_actions = list(reversed(self.prev_actions))
        for i in range(len(self.prev_states)):
            prev_state = prev_states[i]
            prev_action = prev_actions[i]
            if prev_state in self.policy:
                old_value = self.policy[prev_state][prev_action]
            else: old_value = 0
            new_value = (1-self.alpha)*old_value + self.alpha*(self.gamma*(reward))
            self.policy[prev_state][prev_action] = new_value
            reward = self.policy[prev_state][prev_action]
            
    # Train Q-learning agent
    def train_Qlearning_agent(self, iterations):
        
        # Create an agent to compete against
        agent1 = self
        agent2 = Q_learning_player(is_player_1=False)
        agent1.training = True
        agent2.training = True
        
        # Play a new game for each iteration
        for iteration in range(iterations):
            if (iteration % 10) == 0:
                print(self.epsilon)
                self.print_progress_bar(iteration, iterations)
            
            # Save model training progress
            if (iteration % 10000) == 0:
                agent1.policy = merge_policies(agent1.policy, agent2.policy)
                self.save_policy(self.name + str(iteration))
            
            board = Board(dimensions=(7, 6), x_in_a_row=4)
            available_moves = get_available_moves(board)
            while agent1.get_state_reward(board) == None and len(available_moves) > 0:
                
                # play Agent 1's move and update past states
                agent1_move = agent1.get_next_move(board)
                prev_board = board.copy()
                board.push(get_placement(board, agent1_move))
                agent1.prev_states.append(get_board_hash(prev_board))
                agent1.prev_actions.append(agent1_move)
                
                # Check if the game is over
                if board.result() != None:
                    update_policies(board, agent1, agent2)
                    agent1.delete_prev_states()
                    agent2.delete_prev_states()
                    break
                
                # play Agent 2's move and update past states
                available_moves = get_available_moves(board)
                agent2_move = agent2.get_next_move(board)
                prev_board = board.copy()
                board.push(get_placement(board, agent2_move))
                agent2.prev_states.append(get_board_hash(prev_board))
                agent2.prev_actions.append(agent2_move)
                
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
        
        # Check the value of each available move
        for move in available_moves:
            
            # Check if state is in table
            if get_board_hash(board) in Q_table:
                if move in Q_table[get_board_hash(board)]:
                    value = Q_table[get_board_hash(board)][move]
                else:
                    value = 0
                    Q_table[get_board_hash(board)][move] = value
            else:
                value = 0
                Q_table[get_board_hash(board)][move] = value
            
            # Select the highest value move
            if value > max_value:
                max_value = value
                best_move = move
                
            # Epsilon greedy
        if self.training:
            if random.random() < self.epsilon:
                return random.choice(available_moves)
            self.decay_epsilon()
            
        return get_placement(board, best_move)
        
"""# Get a string representation of the board
def get_board_hash(board):
    horizontal_score = check_horizontals(board)
    vertical_score = check_verticals(board)
    diagonal_score = check_diagonals(board)
    heuristic_scores = [horizontal_score, vertical_score, diagonal_score]
    hash = ','.join(map(str, heuristic_scores))
    return hash"""

# Count the number of pieces in each row and column
def get_x_and_o_counts(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    row_counts_x = [0] * num_rows
    row_counts_o = [0] * num_rows
    col_counts_x = [0] * num_cols
    col_counts_o = [0] * num_cols
    
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] == 1:
                row_counts_x[i] += 1
                col_counts_x[j] += 1
            elif matrix[i][j] == 2:
                row_counts_o[i] += 1
                col_counts_o[j] += 1
                
    return row_counts_x, col_counts_x, row_counts_o, col_counts_o

# Get a string representation of the board
def get_board_hash(board):
    row_counts_x, col_counts_x, row_counts_o, col_counts_o = get_x_and_o_counts(board.board)
    hash = ''.join(map(str, row_counts_x + col_counts_x + row_counts_o + col_counts_o))
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

# Merge two Q_learning policies
def merge_policies(policy1, policy2):
    merged_policy = policy1.copy()
    merged_policy.update(policy2)
    return merged_policy

def play_connect_four(board, player1, player2):
    print(f"Game Starting. \nPlayers: {player1.name}, {player2.name}\n")
    player2.is_player_1 = False
    
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

# Print a progress bar
def print_progress_bar(iteration, iterations, bar_length=50):
    progress = iteration/iterations
    arrow = '-' * int(progress * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rPlaying Games: [{arrow + spaces}] {progress:.2%}', end='', flush=True)

# Silently play tictactoe and return the winner (FOR REPORT)
def get_connect_four_winner(board, player1, player2):
    while board.result() == None:
        player1_move = player1.get_next_move(board)
        board.push(player1_move)
        if board.result() != None:
            return board.result()
        player2_move = player2.get_next_move(board)
        board.push(player2_move)
    return board.result()

# Run a number of games and get results (FOR REPORT)
def run_games(player1, player2, num_games):
    results = [0, 0, 0]
    for i in range(num_games):
        board = Board(dimensions=(7, 6), x_in_a_row=4)
        print_progress_bar(i, num_games)
        result = get_connect_four_winner(board, player1, player2)
        results[result] = results[result]+1
        player1.played_moves = 0
        player2.played_moves = 0
    relative_results = [num / num_games for num in results]
    print(f"\nTies: {relative_results[0]}\n{player1.name} wins: {relative_results[1]}\n{player2.name} wins: {relative_results[2]}")
    return relative_results

# Save game results (FOR REPORT)
def write_to_csv(titles, results, filename):
    try:
        with open(filename + '.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(titles)
            writer.writerows(map(str, row) for row in results)
        print("CSV file successfully written:", filename)
    except Exception as e:
        print("Error writing CSV file:", e)

# Run a number of games with Q-learning agents with different training (FOR REPORT)
def test_Q_learning_agents(Q_learning_agent, opponent, num_games):
    results = []
    training_iterations = 0
    Q_learning_agent.is_player_1=True
    opponent.is_player_1=False
    filename = f"CF_Q_learning_agent_P1_vs_{opponent.name}_Results"
    titles = ["Ties", f"{Q_learning_agent.name} wins", f"{opponent.name} wins"]
    for i in range(10):
        print(f"Testing {training_iterations} iterations Q-learning agent")
        policy_name = "Connect_four_Q_learning_agents_100k_iter/Connect_four_Q_learning_agent"+str(training_iterations)
        Q_learning_agent.policy = Q_learning_agent.load_policy(policy_name)
        result = run_games(Q_learning_agent, opponent, num_games)
        results.append(result)
        training_iterations = training_iterations+10000
    write_to_csv(titles, results, filename)
    print(results)
    
    # Switch player order
    results = []
    training_iterations = 0
    Q_learning_agent.is_player_1=False
    opponent.is_player_1=True
    filename = f"CF_Q_learning_agent_P2_vs_{opponent.name}_Results"
    titles = ["Ties", f"{opponent.name} wins", f"{Q_learning_agent.name} wins"]
    for i in range(10):
        print(f"Testing {training_iterations} iterations Q-learning agent")
        policy_name = "Connect_four_Q_learning_agents_100k_iter/Connect_four_Q_learning_agent"+str(training_iterations)
        Q_learning_agent.policy = Q_learning_agent.load_policy(policy_name)
        result = run_games(opponent, Q_learning_agent, num_games)
        results.append(result)
        training_iterations = training_iterations+10000
    write_to_csv(titles, results, filename)
    print(results)
    
    # Run a number of games with agents(FOR REPORT)
def test_agents(agent, opponent, num_games):
    results = []
    agent.is_player_1=True
    opponent.is_player_1=False
    filename = f"Minimax_agent_P1_vs_{opponent.name}_Results"
    titles = ["Ties", f"{agent.name} wins", f"{opponent.name} wins"]
    result = run_games(agent, opponent, num_games)
    results.append(result)
    write_to_csv(titles, results, filename)
    print(results)
    
    # Switch player order
    results = []
    agent.is_player_1=False
    opponent.is_player_1=True
    filename = f"Minimax_agent_P2_vs_{opponent.name}_Results"
    titles = ["Ties", f"{opponent.name} wins", f"{agent.name} wins"]
    result = run_games(opponent, agent, num_games)
    results.append(result)
    write_to_csv(titles, results, filename)
    print(results)
                
tictactoe_board = Board(dimensions=(7, 6), x_in_a_row=4)
#print(get_available_moves(tictactoe_board))
default = Default_player(optimality = .5)
human = Human_player()
rand = Random_player()
minimax = Minimax_player()
#playa1 = Q_learning_player(policy_name="Connect_Four_Q_learning_agent")
qlearning = Q_learning_player()#(policy_name="Connect_four_Q_learning_agents/Connect_Four_Q_learning_agent9000")
qlearning.train_Qlearning_agent(100001)

test_Q_learning_agents(qlearning, default, 1000)
#test_agents(minimax, default, 1000)

play_connect_four(tictactoe_board, qlearning, default)
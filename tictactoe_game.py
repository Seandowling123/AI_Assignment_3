from tictactoe import Board
from collections import defaultdict
import pickle
import math
import random
import csv
import os

agent1_wins = 0

# Get all available moves
def get_available_moves(board):
    zero_indices = []
    for i, row in enumerate(board.board):
        for j, value in enumerate(row):
            if value == 0:
                zero_indices.append((i, j))
    return zero_indices

class Default_player:
    def __init__(self, optimality = 1, is_player_1 = True):
        self.name = "Default Player"
        self.optimality = optimality
        self.is_player_1 = is_player_1
        
    # Get the reward for taking an action
    def get_state_reward(self, board_state):
        is_player_1 = self.is_player_1
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
        
        if len(available_moves) == 9:
            return [1,1]
        
        # Check for winning moves
        for move in available_moves:
            new_board = board.copy()
            new_board.push(move)
            result = self.get_state_reward(new_board)
            if result > 0 and optimal:
                return move
        
        # Block loosing moves
        for move in available_moves:
            new_board = board.copy()
            if self.is_player_1:
                new_board.turn = 2
            else: new_board.turn = 1
            new_board.push(move)
            result = self.get_state_reward(new_board)
            if result < 0 and optimal:
                return move
        return random.choice(available_moves)

class Random_player:
    def __init__(self):
        self.name = "Random Player"
        
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = random.choice(available_moves)
        return move

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
                move_input = input("Enter a row and column:").split(' ')
                row = int(move_input[1])
                col = int(move_input[0])
                move = [row, col]
                if move in available_moves:
                    return move
                else: print("Invalid Move.")
            except Exception as e:
                print("Invalid Move.")

# Player using minimax
class Minimax_player:
    def __init__(self, is_player_1=True):
        self.name = "Minimax"
        self.is_player_1 = is_player_1
        
    # Get the reward for taking an action
    def get_state_reward(self, board_state):
        is_player_1 = self.is_player_1
        result = board_state.result()
        if result == 1 and is_player_1:
            return None, 1
        elif result == 1 and not is_player_1:
            return None, -1
        elif result == 2 and is_player_1:
            return None, -1
        elif result == 2 and not is_player_1:
            return None, 1
        return None, 0
    
    # Use minimax algorithm to decide the next move
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first move
        available_moves = get_available_moves(board)
        if len(available_moves) == 9:
            return [1,1], 0
        elif len(available_moves) == 8:
            if [0,0] in available_moves:
                return [0,0], 0
            else: return [1,1], 0
        
        # Check if the game is over
        if board.result() != None:
            return self.get_state_reward(board)
        
        # Calculate move for maximiser
        if is_maximising:
            max_value = -math.inf
            best_move = None
            
            # get value of each move
            for move in available_moves:
                new_board = board.copy()
                new_board.push(move)
                best_move, value = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value >= max_value:
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
                new_board.push(move)
                best_move, value = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value <= min_value:
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
        return move[0]
    
class Q_learning_player:
    def __init__(self, policy_name=None, alpha=.2, gamma=.9, epsilon=.3, training=False, is_player_1=True):
        self.name = "Tictactoe_Q_learning_agent"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = .00001
        self.final_epsilon = .0001
        self.policy_name = policy_name
        self.training = training
        self.is_player_1 = is_player_1
        self.prev_states = []
        self.prev_actions = []
        if policy_name != None:
            self.policy = self.load_policy(policy_name)
        else: self.policy = defaultdict(dict)
    
    # Save Qtable
    def save_policy(self, policy_name):
        filehandler = open("Tictactoe_Q_learning_agents/"+policy_name,"wb")
        pickle.dump(self.policy, filehandler)
        filehandler.close()
    
    # Load a Qtable
    def load_policy(self, policy_name):
        if os.path.exists(policy_name):
            file = open(policy_name, 'rb')
            policy = pickle.load(file)
            file.close()
            return policy
        else:
            print("Error: File does not exist")
            return None
    
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
        result = board_state.result()
        if result == 1 and is_player_1:
            return 1
        elif result == 1 and not is_player_1:
            return -1
        elif result == 2 and is_player_1:
            return -1
        elif result == 2 and not is_player_1:
            return 1
        elif result == 0:
            return 0
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
            if (iteration % 100) == 0:
                print(f" Epsilon: {self.epsilon}", end='', flush=True)
                self.print_progress_bar(iteration, iterations)
                
            # Save model training progress
            if (iteration % 10000) == 0:
                agent1.policy = merge_policies(agent1.policy, agent2.policy)
                self.save_policy(self.name + str(iteration))
            
            # PLay each game
            board = Board(dimensions=(3, 3))
            available_moves = get_available_moves(board)
            while agent1.get_state_reward(board) == None and len(available_moves) > 0:
                
                # play Agent 1's move and update past states
                agent1_move = agent1.get_next_move(board)
                prev_board = board.copy()
                board.push(agent1_move)
                agent1.prev_states.append(get_board_hash(prev_board))
                agent1.prev_actions.append(agent1_move)
                
                # Check if the game is over
                if board.result() != None:
                    update_policies(board, agent1, agent2)
                    agent1.delete_prev_states()
                    agent2.delete_prev_states()
                    break
                
                # play Agent 2's move and update past states
                agent2_move = agent2.get_next_move(board)
                prev_board = board.copy()
                board.push(agent2_move)
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
            
        return best_move

# Get a string representation of the board
def get_board_hash(board):
    hash = ''.join(map(str, board.board.flatten()))
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

# Combine two Q-learning policies
def merge_policies(policy1, policy2):
    merged_policy = policy1.copy()
    merged_policy.update(policy2)
    return merged_policy

# Play a game of tictactoe
def play_tictactoe(board, player1, player2):
    player2.is_player_1 = False
    print(f"Game Starting. \nPlayers: {player1.name}, {player2.name}\n")
    
    # While the game is not over let each player move
    while board.result() == None:
        player1_move = player1.get_next_move(board)
        board.push(player1_move)
        print(board, "\n")
        if board.result() != None:
            break
        player2_move = player2.get_next_move(board)
        board.push(player2_move)
        print(board, "\n")
        
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
def get_tictactoe_winner(board, player1, player2):
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
        board = Board(dimensions=(3, 3))
        print_progress_bar(i, num_games)
        result = get_tictactoe_winner(board, player1, player2)
        results[result] = results[result]+1
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

tictactoe_board = Board(dimensions=(3, 3))
human = Human_player()
minimax = Minimax_player()
rand = Random_player()
default = Default_player(optimality=.5)
qlearning = Q_learning_player(policy_name="Tictactoe_Q_learning_agents/Tictactoe_Q_learning_agent80000")
#qlearning.train_Qlearning_agent(100000)
play_tictactoe(tictactoe_board, default, qlearning)
#results = run_games(minimax, default, 1000)

#test_agents(minimax, default, 1000)

# (training=True)#

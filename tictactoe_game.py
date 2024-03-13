from tictactoe import Board
import pickle
import math
import random

agent1_wins = 0

# Get all available moves
def get_available_moves(board):
    zero_indices = []
    for i, row in enumerate(board.board):
        for j, value in enumerate(row):
            if value == 0:
                zero_indices.append([i, j])
    return zero_indices

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
    def __init__(self):
        self.name = "Minimax"
        
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first move
        available_moves = get_available_moves(board)
        if len(available_moves) == 9:
            return [1,1], 0
        
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
                new_board.push(move)
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
        #print(move)
        return move[0]
    
class Q_learning_player:
    def __init__(self, policy_name=None, alpha=.2, gamma=.9, is_player_1=True):
        self.name = "Tictactoe_Q_learning_agent"
        self.alpha = alpha
        self.gamma = gamma
        self.policy_name = policy_name
        self.is_player_1 = is_player_1
        self.prev_states = []
        if policy_name != None:
            self.policy = self.load_policy(policy_name)
        else: self.policy = {}
    
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
                
            board = Board(dimensions=(3, 3))
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
        
        # Check the value of each available move
        for move in available_moves:
            new_board = board.copy()
            new_board.push(move)
            
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

def merge_policies(policy1, policy2):
    merged_policy = policy1.copy()
    merged_policy.update(policy2)
    return merged_policy
        
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

tictactoe_board = Board(dimensions=(3, 3))
#player1 = Minimax_player()
playa1 = Random_player()
playa2 = Q_learning_player(policy_name="Tictactoe_Q_learning_agent", is_player_1=False)
#playa2.train_Qlearning_agent(10000)

play_tictactoe(tictactoe_board, playa1, playa2)
from tictactoe import Board
import pickle
import math
import random

# board.possible_moves()
# board.board

# Get all available moves
def get_available_moves(board):
    zero_indices = []
    for i, row in enumerate(board.board):
        for j, value in enumerate(row):
            if value == 0:
                zero_indices.append([i, j])
    return zero_indices

class random_player:
    def __init__(self):
        self.name = "Random Player"
        
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = random.choice(available_moves)
        return move

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
class minimax_player:
    def __init__(self):
        self.name = "Minimax"
        
    def minimax(self, board, alpha, beta, is_maximising, depth):
        
        # Speed up for first move
        available_moves = get_available_moves(board)
        if len(available_moves) == 9:
            return 0, [1,1]
        
        # Check if the game is over
        try:
            result = board.result()
            if result == 1:
                return 1, None
            elif result == 2:
                return -1, None
        except Exception as e:
            if str(e) == "Both X and O have 3 pieces in a row.":
                return 0, None
            else: print(f"A minimax Error Occured: {e}")
        
        # Calculate move for maximiser
        if is_maximising:
            max_value = -math.inf
            best_move = None
            
            # get value of each move
            for move in available_moves:
                new_board = board.copy()
                new_board.push(move)
                value, best_move = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value >= max_value:
                    max_value = value
                    best_move = move
                
                # Prune search
                if max_value > alpha:
                    alpha = max_value
                if alpha >= beta:
                    break
                    
            return max_value, best_move
        
        # Calculate move for minimiser
        else:
            min_value = math.inf
            best_move = None
            
            # Get value of each move
            for move in available_moves:
                new_board = board.copy()
                new_board.push(move)
                value, best_move = self.minimax(new_board, alpha, beta, not is_maximising, depth+1)
                
                # Find the best move
                if value <= min_value:
                    min_value = value
                    best_move = move
                
                # Prune search
                if min_value < beta:
                    beta = min_value
                if alpha >= beta:
                    break
                
            return min_value, best_move
         
    # Return the next move for the player
    def get_next_move(self, board):
        move = self.minimax(board, -math.inf, math.inf, True, 0)
        print(f"Minimax move: {move[1][1]}, {move[1][0]}.")
        return move[1]
    
class Qlearning_player:
    def __init__(self, policy_name=None, alpha=.2, gamma=.9):
        self.name = "Minimax"
        self.alpha = alpha
        self.gamma = gamma
        self.policy_name = policy_name
        if policy_name != None:
            self.policy = self.load_policy(policy_name)
        else: self.policy = {}
    
    # Save Qtable
    def save_policy(policy, policy_name):
        filehandler = open(policy_name,"wb")
        pickle.dump(policy,filehandler)
        filehandler.close()
    
    # Load a Qtable
    def load_policy(policy_name):
        file = open(policy_name,'rb')
        policy = pickle.load(file)
        file.close()
        return policy
    
    def get_board_hash(board):
        hash = str(board.board.flatten())
        return hash
    
    # Get the reward for taking an action
    def get_state_reward(board_state):
        try:
            result = board_state.result()
            if result == 2:
                return 1
            elif result == 1:
                return -1
        except Exception as e:
            if str(e) == "Both X and O have 3 pieces in a row.":
                return 0
            else: print(f"A Q Learning state reward Error Occured: {e}")
        return None

    # Update Q-values 
    def update_policy(self, state, value):
        old_value = self.policy[state]
        new_value = (1-self.alpha)*old_value + self.alpha*(self.gamma*(value))
        self.policy[self.get_board_hash(state)] = new_value
    
    # Train the agent
    def train(self, iterations):
        Q_table = self.policy
        for i in range(iterations):
            
            board = Board(dimensions=(3, 3))
            while self.get_state_reward(board) != None:
                print(board)
                    
                value = None
                max_value = -math.inf
                best_move = None

                # Get the best move from the policy
                available_moves = get_available_moves(board)
                for move in available_moves:
                    new_board = board.copy()
                    new_board.push(move)
                    
                    # Check if move leads to a terminal state
                    if self.get_state_reward(new_board) != None:
                        value = self.get_state_reward(new_board)
                        
                    # Check if state is in table
                    elif self.get_board_hash(new_board) in Q_table:
                        value = Q_table[self.get_board_hash(new_board)]
                    else:
                        value = 0
                        Q_table[self.get_board_hash(new_board)] = value
                    
                    # Execute the highest value move
                    if value > max_value:
                        max_value = value
                        best_move = move
                        
                # Update policy 
                self.update_policy(board, max_value)
                board.push(best_move)
                
    
    def get_next_move(self, board):
        value = -math.inf
        Q_table = self.policy
        available_moves = get_available_moves(board)
        for move in available_moves:
            new_board = board.copy()
            new_board.push(move)
            
            # Check if move leads to a terminal state
            if self.get_state_reward(new_board) != None:
                value = self.get_state_reward(new_board)
                
            # Check if state is in table
            elif move in Q_table:
                value = Q_table[move]
            else:
                value = 0
                
        
        
     
def play_tictactoe(board, player1, player2):
    
    # While the game is not over let each player move
    try:
        while board.result() == None:
            player1_move = player1.get_next_move(board)
            board.push(player1_move)
            if board.result():
                print(board)
                break
            player2_move = player2.get_next_move(board)
            board.push(player2_move)
            print(board)
            
        # Print the winner
        if board.result() == 1:
            print(f"Winner = {player1.name}")
        else: print(f"Winner = {player2.name}")
    except Exception as e:
        if str(e) == "Both X and O have 3 pieces in a row.":
            print("Tie Game")
        else: print(e)

tictactoe_board = Board(dimensions=(3, 3))
#playa1 = minimax_player()
#playa2 = random_player()
player2 = Qlearning_player()
player2.train(10)

#play_tictactoe(tictactoe_board, playa1, playa2)
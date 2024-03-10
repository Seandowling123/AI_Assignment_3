from tictactoe import Board
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

# Human contolled player
class human_player:
    def __init__(self, name):
        self.name = name
    
    def get_next_move(self, board):
        available_moves = get_available_moves(board)
        move = None
        
        while move not in available_moves or move == None:
            try:
                move_input = input("Enter a row and column:").split(' ')
                row = int(move_input[1])
                col = int(move_input[0])
                move = [row, col]
                if move in available_moves:
                    return move
            except Exception as e:
                print("Invalid Move.")

# Player using minimax
class minimax_player:
    def __init__(self, name):
        self.name = name
        
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
        
        # Calculate value of moves
        if is_maximising:
            #print("maximising")
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
        
        else:
            #print("minimising")
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
        print(move)
        print(f"Minimax move: {move[1]}, {move[0]}.")
        return move[1]
        
     
def play_tictactoe(board, player1, player2):
    
    # While the game is not over let each player move
    try:
        while board.result() == None:
            player1_move = player1.get_next_move(board)
            print(player1_move)
            board.push(player1_move)
            player2_move = player2.get_next_move(board)
            board.push(player2_move)
            print(board)
        
        if board.result() == 1:
            print(f"Winner = {player1.name}")
        else: print(f"Winner = {player2.name}")
    except Exception as e:
        if str(e) == "Both X and O have 3 pieces in a row.":
            print("Tie Game")
        else: print(e)

tictactoe_board = Board(dimensions=(3, 3))
playa1 = minimax_player("Minimax")
#playa1 = human_player("1")
playa2 = human_player("Sean")

play_tictactoe(tictactoe_board, playa1, playa2)
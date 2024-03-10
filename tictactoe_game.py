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
    
    def get_next_move(self, possible_moves):
        move_input = input("Enter a row and column:").split(' ')
        row = int(move_input[0])
        col = int(move_input[1])
        move = [row, col]
        if move in possible_moves:
            return move

# Player using minimax
class minimax_player:
    def __init__(self, name):
        self.name = name
        
    def minimax(self, board, alpha, beta, maximising):
        # Check if the game is over
        result = board.result()
        if result == 1:
            return 1, None
        elif result == 1:
            return -1, None
        elif result == 0:
            return 0, None
        
        available_moves = get_available_moves(board)
        if maximising:
            max_value = -math.inf
            best_move = None
            
            # Check moves
            for move in available_moves:
                new_board = board.copy()
                new_board.push(move)
                value, best_move = self.minimax(new_board, alpha, beta, not maximising)
                
                if value > max_value:
                    max_value = value
                    best_move = move
            
                
                
            
    
    def get_next_move(self, board, possible_moves):
        
        
     
def play_tictactoe(board, player1, player2):
    
    # While the game is not over let each player move
    while board.result() == None:
        available_moves = get_available_moves(board)
        player1_move = player1.get_next_move(available_moves)
        board.push(player1_move)
        player2_move = player2.get_next_move(available_moves)
        board.push(player2_move)
        print(board)
    
    print(f"Winner = {board.result()}")

tictactoe_board = Board(dimensions=(3, 3))
playa1 = human_player("1")
playa2 = human_player("2")

play_tictactoe(tictactoe_board, playa1, playa2)
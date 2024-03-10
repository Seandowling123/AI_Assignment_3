from tictactoe import Board
from tictactoe.egtb import Reader
import random

# board.possible_moves()
# board.board.flatten()

def get_available_moves(board):
    zero_indices = []
    for i, row in enumerate(board.board):
        for j, value in enumerate(row):
            if value == 0:
                zero_indices.append([i, j])
    return zero_indices

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
        
def play_tictactoe(board, player1, player2):
    while board.result() == None:
        available_moves = get_available_moves(board)
        print(available_moves)
        player1_move = player1.get_next_move(available_moves)
        print(player1_move)
        board.push(player1_move)
        player2_move = player2.get_next_move(available_moves)
        board.push(player2_move)
        print(board)

tictactoe_board = Board(dimensions=(3, 3))
playa1 = human_player("1")
playa2 = human_player("2")

play_tictactoe(tictactoe_board, playa1, playa2)
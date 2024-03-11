from tictactoe import Board
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
tictactoe_board.push(get_placement(tictactoe_board, 3))
print(get_available_moves(tictactoe_board))
player1 = human_player()
playa2 = random_player()
#playa2 = Qlearning_player(policy_name='Q_learning_agent')
#player1.train_Qlearning_agent(10000)

play_tictactoe(tictactoe_board, player1, playa2)
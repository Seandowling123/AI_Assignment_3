def print_diagonals(board):
    rows = len(board)
    cols = len(board[0])

    # Iterate over diagonals starting from bottom-left corner
    for i in range(rows + cols - 1):
        diagonal = []
        for j in range(max(0, i - cols + 1), min(rows, i + 1)):
            diagonal.append(board[j][i - j])
        print(diagonal)

    # Iterate over diagonals starting from bottom-right corner
    for i in range(rows + cols - 1):
        diagonal = []
        for j in range(max(0, i - cols + 1), min(rows, i + 1)):
            diagonal.append(board[j][cols - i + j - 1])
        print(diagonal)

# Example usage:
board = [
    ['1', '2', '3', '4'],
    ['5', '6', '7', '8'],
    ['9', '10', '11', '12'],
    ['13', '14', '15', '16']
]
print_diagonals(board)

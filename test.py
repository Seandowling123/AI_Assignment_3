# Original 7x6 matrix
original_matrix = [
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28],
    [29, 30, 31, 32, 33, 34, 35],
    [36, 37, 38, 39, 40, 41, 42]
]

# Initialize the transposed matrix (6x7) with zeros
transposed_matrix = [[0 for _ in range(len(original_matrix))] for _ in range(len(original_matrix[0]))]

# Transpose the matrix
for i in range(len(original_matrix)):
    for j in range(len(original_matrix[0])):
        transposed_matrix[j][i] = original_matrix[i][j]

# Output the transposed matrix (6x7)
for row in transposed_matrix:
    print(row)

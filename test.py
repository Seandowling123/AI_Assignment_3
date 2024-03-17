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

matrix = [
    [1, 0, 2, 1],
    [2, 1, 1, 0],
    [0, 1, 2, 2]
]

row_counts_x, col_counts_x, row_counts_o, col_counts_o = get_x_and_o_counts(matrix)
print(row_counts_x, col_counts_x, row_counts_o, col_counts_o)
combined_counts = ''.join(map(str, row_counts_x + col_counts_x + row_counts_o + col_counts_o))
print("Combined Counts:", combined_counts)

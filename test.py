def count_1s_and_2s_separately(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    row_counts_1 = [0] * num_rows
    row_counts_2 = [0] * num_rows
    col_counts_1 = [0] * num_cols
    col_counts_2 = [0] * num_cols
    
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] == 1:
                row_counts_1[i] += 1
                col_counts_1[j] += 1
            elif matrix[i][j] == 2:
                row_counts_2[i] += 1
                col_counts_2[j] += 1
                
    return row_counts_1, row_counts_2, col_counts_1, col_counts_2

matrix = [
    [1, 0, 2, 1],
    [2, 1, 1, 0],
    [0, 1, 2, 2]
]

row_counts_1, row_counts_2, col_counts_1, col_counts_2 = count_1s_and_2s_separately(matrix)

combined_counts = ' '.join(map(str, row_counts_1 + row_counts_2 + col_counts_1 + col_counts_2))
print("Combined Counts:", combined_counts)

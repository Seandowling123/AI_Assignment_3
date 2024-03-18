# Example nested dictionary
nested_dict = {
    'A': {'x': 10, 'y': 20, 'z': 30},
    'B': {'x': 15, 'y': 25, 'z': 35},
    'C': {'x': 25, 'y': 35, 'z': 45}
}

# Find the maximum value
max_value = max(max(inner_dict.values()) for inner_dict in nested_dict.values())

print("Max value:", max_value)

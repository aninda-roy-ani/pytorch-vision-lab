import torch

# Create a 3x4 tensor
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)

# Get the first two rows
first_two_rows = x[:2]

print("\nSLICING FIRST TWO ROWS ([0:2]):\n\n", first_two_rows)
print("-" * 55)

# Get the third column of all rows
third_column = x[:, 2]

print("\nSLICING THIRD COLUMN ([:, 2]):", third_column)
print("-" * 55)

# Every other column
every_other_col = x[:, ::2]

print("\nEVERY OTHER COLUMN ([:, ::2]):\n\n", every_other_col)
print("-" * 55)

# Last column
last_col = x[:, -1]

print("\nLAST COLUMN ([:, -1]):", last_col, "\n")

# Combining slicing and indexing (First two rows, last two columns)
combined = x[:2, 2:]

print("\nFIRST TWO ROWS, LAST TWO COLS ([0:2, 2:]):\n\n", combined, "\n")


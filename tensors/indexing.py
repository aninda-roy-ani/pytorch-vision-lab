import torch

# Create a 3x4 tensor
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
print("ORIGINAL TENSOR:\n\n", x)
print("-" * 55)


# Get a single element at row 1, column 2
single_element_tensor = x[1, 2]

print("\nINDEXING SINGLE ELEMENT AT [1, 2]:", single_element_tensor)
print("-" * 55)

# Get the entire second row (index 1)
second_row = x[1]

print("\nINDEXING ENTIRE ROW [1]:", second_row)
print("-" * 55)

# Last row
last_row = x[-1]

print("\nINDEXING ENTIRE LAST ROW ([-1]):", last_row, "\n")

print("SINGLE-ELEMENT TENSOR:", single_element_tensor)
print("-" * 45)

# Extract the value from a single-element tensor as a standard Python number
value = single_element_tensor.item()

print("\n.item() PYTHON NUMBER EXTRACTED:", value)
print("TYPE:", type(value))

# Boolean indexing using logical comparisons
mask = x > 6

print("MASK (VALUES > 6):\n\n", mask, "\n")

# Applying Boolean masking
mask_applied = x[mask]

print("VALUES AFTER APPLYING MASK:", mask_applied, "\n")

# Fancy indexing

# Get first and third rows
row_indices = torch.tensor([0, 2])

# Get second and fourth columns
col_indices = torch.tensor([1, 3]) 

# Gets values at (0,1), (0,3), (2,1), (2,3)
get_values = x[row_indices[:, None], col_indices]

print("\nSPECIFIC ELEMENTS USING INDICES:\n\n", get_values, "\n")
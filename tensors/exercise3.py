"""
Exercise 3: Combining and Weighting Sensor Data
You're building an environment monitoring system that uses two sensors: one for temperature and one for humidity. 
You receive data from these sensors as two separate 1D tensors.

Your task is to:

Concatenate the two tensors into a single 2x5 tensor, where the first row is temperature data and the second is humidity data.
Create a weights tensor torch.tensor([0.6, 0.4]).
Use broadcasting and element-wise multiplication to apply these weights to the combined sensor data. The temperature data should be 
multiplied by 0.6 and the humidity data by 0.4.
Finally, calculate the weighted average for each time step by summing the weighted values along dim=0 and dividing by 
the sum of the weights.

"""

import torch

# Sensor readings (5 time steps)
temperature = torch.tensor([22.5, 23.1, 21.9, 22.8, 23.5])
humidity = torch.tensor([55.2, 56.4, 54.8, 57.1, 56.8])

print("TEMPERATURE DATA: ", temperature)
print("HUMIDITY DATA:    ", humidity)
print("-" * 45)

### START CODE HERE ###

# 1. Concatenate the two tensors.
# Note: You need to unsqueeze them first to stack them vertically.
combined_data = torch.cat(tensors=(temperature,humidity)).reshape(2,5)

# 2. Create the weights tensor.
weights = torch.tensor([0.6, 0.4])

# 3. Apply weights using broadcasting.
# You need to reshape weights to [2, 1] to broadcast across columns.
weighted_data = combined_data * weights.reshape(2,1)

# 4. Calculate the weighted average for each time step.
#    (A true average = weighted sum / sum of weights)
weighted_sum = sum(weighted_data)
weighted_average = weighted_sum/ sum(weights)

### END CODE HERE ###

print("\nCOMBINED DATA (2x5):\n\n", combined_data)
print("\nWEIGHTED DATA:\n\n", weighted_data)
print("\nWEIGHTED AVERAGE:", weighted_average)
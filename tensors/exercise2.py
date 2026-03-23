"""
Exercise 2: Image Batch Transformation
You're working on a computer vision model and have a batch of 4 grayscale images, each with a size of 3 pixels high by 2 pixels wide. 
The data is currently in a tensor with the shape [4, 3, 2], which represents [batch_size, height, width].

For processing with certain deep learning frameworks, you need to transform this data into the [batch_size, channels, height, width] format.
 Since the images are grayscale, you'll need to:

Add a new dimension of size 1 at index 1 to represent the color channel.
After adding the channel, you realize the model expects the shape [batch_size, height, width, channels]. 
Transpose the tensor to move the channel dimension to the last position without scrambling the height and width. 
Use .transpose() multiple times.
"""

import torch

# A batch of 4 grayscale images, each 3x2
image_batch = torch.rand(4, 3, 2)

print("ORIGINAL BATCH SHAPE:", image_batch.shape)
print("-" * 45)
print("ORIGINAL BATCH:", image_batch)
print("-" * 45)

### START CODE HERE ###

# 1. Add a channel dimension at index 1.
# Result: [batch, channels, height, width]
image_batch_with_channel = image_batch.unsqueeze(1)
print("\nSHAPE AFTER UNSQUEEZE:", image_batch_with_channel.shape)
print("WITH CHANNEL BATCH:", image_batch_with_channel)
print("-" * 45)

# 2. Transpose to get [batch, height, width, channels].
# Step A: Swap channels (1) with height (2)
# Result: [batch, height, channels, width]
temp_tensor = image_batch_with_channel.transpose(1,2)
print("AFTER FIRST TRANSPOSE BATCH:", temp_tensor)
print("-" * 45)

# Step B: Swap channels (2) with width (3)
# Result: [batch, height, width, channels]
image_batch_transposed = temp_tensor.transpose(2,3)
print("SHAPE AFTER TRANSPOSE:", image_batch_transposed.shape)
print("FINAL BATCH:", image_batch_transposed)
print("-" * 45)

### END CODE HERE ###

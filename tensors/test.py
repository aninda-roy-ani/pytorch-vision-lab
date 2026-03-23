import torch
import torch.nn as nn



single_distance = torch.tensor(25.0)
print(single_distance)
print(single_distance.shape)

dim1 = single_distance.unsqueeze(0)
print(dim1)
print(dim1.shape)

dim2 = dim1.unsqueeze(1)
print(dim2)
print(dim2.shape)

dim3 = dim2.squeeze()
print(dim3)
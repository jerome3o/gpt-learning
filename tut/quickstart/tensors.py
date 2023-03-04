import torch
import numpy as np

# tensors are similar to np arrays but can be put on gpus, optimised for differentiation

# initialising a tensor

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy
# on cpu device numpy arrays and tensors share underlying memory

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensors
# retain the properties of x_data
x_ones = torch.ones_like(x_data) 
print(f"Ones Tensor: \n {x_ones} \n")

# override the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor: \n {x_rand} \n")

# with random or constant values:
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Attributes, (shape, dtype, device)
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on tensors
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# most other operations are similar to numpy
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Joining tensors

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic 
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In place operations, denoted by _ suffix
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)
# usage of inplace is discouraged, as the immediate loss of history is problematic for derivative calculation

# Bridge with numpy
# on cpu they share underlying memory

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy to tensor

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

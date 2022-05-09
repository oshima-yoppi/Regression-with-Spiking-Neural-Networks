import torch
# a = torch.tensor([[1,0.1],[2, 0.2]])

# b = torch.tensor([[3,0.3],[4, 0.4]])
# c = torch.stack((a,b), 1)
# print(a-1)
# print(b)
# for i in range(2):
#     print(c[:,i])
# # print(c)
a = torch.tensor([1,2])
b = a.unsqueeze(0)
print(b)
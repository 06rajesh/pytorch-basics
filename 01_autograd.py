import torch

# x = torch.randn(3, requires_grad=True)
#
# print(x)
#
# y = x+2
# print(y)
# z = y*y*2
# print(z)
#
# v = torch.tensor([1.0, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)  # dz/dx
# print(x.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    out = (weights*3).sum()
    out.backward()
    print(weights.grad)
    weights.grad.zero_()

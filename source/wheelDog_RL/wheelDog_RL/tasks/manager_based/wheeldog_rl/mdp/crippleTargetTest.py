import torch

delta = torch.Tensor([[2, 2], [4, 4]])
bravo = torch.Tensor([2, 2]).unsqueeze(0)

print(f"delta - bravo: {delta - bravo}")
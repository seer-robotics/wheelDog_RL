import torch

scale_levels = 8
min_factor = 1e-2

mean_levels = torch.arange(0, 20.5, 0.5)
# mean_levels = torch.mean(torch.Tensor([10, 11]))

stage = ((torch.floor(mean_levels)) * scale_levels) // 20
stage = torch.clamp(stage, min=0, max = scale_levels-1)

factor = 1.0 + (min_factor - 1.0) * (stage/(scale_levels-1))

print(f"mean_levels: \n{mean_levels}")
print(f"mean_levels.shape: \n{mean_levels.shape}")
print(f"stages: \n{factor}")
print(f"stages.shape: \n{factor.shape}")

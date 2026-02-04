import torch

action_levels = 8
max_scale = 2.0

mean_levels = torch.arange(0, 20.5, 0.5)
# mean_levels = torch.mean(torch.Tensor([10, 11]))

stage = ((torch.floor(mean_levels)) * action_levels) // 20
stage = torch.clamp(stage, min=0, max = action_levels-1)

factor = 1.0 + (max_scale - 1.0) * (stage/(action_levels-1))

print(f"mean_levels: \n{mean_levels}")
print(f"mean_levels.shape: \n{mean_levels.shape}")
print(f"stages: \n{factor}")
print(f"stages.shape: \n{factor.shape}")
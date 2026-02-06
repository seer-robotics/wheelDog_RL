import torch

scale_levels = 4
min_factor = 0.1
min_factor_terrainLevel: int = 4

mean_levels = torch.arange(0, 20.5, 0.5)
# mean_levels = torch.mean(torch.Tensor([10, 11]))

stage = (torch.clamp(torch.floor(mean_levels), min=0, max=min_factor_terrainLevel) * scale_levels) // min_factor_terrainLevel
stage = torch.clamp(stage, min=0, max = scale_levels-1)

factor = 1.0 + (min_factor - 1.0) * (stage/(scale_levels-1))

print(f"mean_levels: \n{mean_levels}")
print(f"mean_levels.shape: \n{mean_levels.shape}")
print(f"stages: \n{factor}")
print(f"stages.shape: \n{factor.shape}")

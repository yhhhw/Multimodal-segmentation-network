import torch
import torch.nn as nn

from wave_branch import WTM
from convolution_branch import Block as ConvNextBlock


class HybridAttention(nn.Module):
    def __init__(self, 
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 d_embedding=512,
                 depth=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.con = nn.Sequential(*[ConvNextBlock(dim=dim, drop_path=0.3, layer_scale_init_value=1.0) for j in range(depth)])

        inner_dim = max(16, dim//reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)
        
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.wave = WTM(d_model=dim, d_k=1, d_v=1, n_head=1, d_embedding=d_embedding)
        

    def forward(self, x1, x2):
        x1 = self.con(x1)
        x2 = self.wave(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.proj(x) + x
        return x


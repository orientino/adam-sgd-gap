"""
Vision Transformer Small (ViT-S/16) for ImageNet-1K.
https://arxiv.org/abs/2205.01580
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def posemb_sincos_2d(h, w, width, temperature=10_000.0):
    """https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py#L34"""
    y, x = np.mgrid[:h, :w]
    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = np.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = np.einsum("m,d->md", y.flatten().astype(np.float32), omega)
    x = np.einsum("m,d->md", x.flatten().astype(np.float32), omega)
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_embed=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, d_embed, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(self.d_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.d_head, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm_q(q), self.norm_k(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_layers=12,
        n_heads=6,
        d_embed=384,
        n_classes=1000,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_embed = d_embed

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            d_embed=d_embed,
        )
        grid_size = self.patch_embed.grid_size
        pos_embed = posemb_sincos_2d(grid_size, grid_size, d_embed)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.register_buffer("pos_embed", pos_embed)
        self.norm_embed = nn.RMSNorm(d_embed, eps=1e-6)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(d_embed, eps=1e-6)
        self.head = nn.Linear(d_embed, n_classes, bias=False)
        self._init_weights()

    def _init_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            block.attn.proj.weight.data *= scale
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            block.mlp.fc2.weight.data *= scale
        nn.init.zeros_(self.head.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.norm_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


def vit_small_patch16_224(n_layers=6, n_heads=6, d_embed=384, n_classes=1000):
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embed=d_embed,
        n_classes=n_classes,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    model = vit_small_patch16_224(n_layers=6, n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Positional embedding shape: {model.pos_embed.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

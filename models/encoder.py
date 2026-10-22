import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.transforms import NUM_BANDS


class PatchEmbed(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int, band_adapter: nn.Module = None):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.band_adapter = band_adapter  # None, or a BandAdapter mapping NUM_BANDS -> in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, NUM_BANDS, H, W) -> (B, N, D)
        if self.band_adapter is not None:
            x = self.band_adapter(x)
        x = self.proj(x)  # (B, D, H/p, W/p)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, (h, w)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        out = self.proj_drop(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 4 == 0
    dim_each = embed_dim // 2

    def emb_1d(pos, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_w = emb_1d(grid[0], dim_each)
    emb_h = emb_1d(grid[1], dim_each)
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (grid*grid, embed_dim)
    return pos_embed.unsqueeze(0)  # (1, N, D)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = NUM_BANDS,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        pos = sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos, persistent=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pos_embed_for(self, grid_size: int) -> torch.Tensor:
        if grid_size == self.grid_size:
            return self.pos_embed
        return sincos_pos_embed(self.embed_dim, grid_size).to(self.pos_embed.device)

    def forward_full(self, x: torch.Tensor, return_all_layers: bool = False):
        tokens, (h, w) = self.patch_embed(x)
        tokens = tokens + self._pos_embed_for(h)
        all_layers = []
        for blk in self.blocks:
            tokens = blk(tokens)
            if return_all_layers:
                all_layers.append(tokens)
        tokens = self.norm(tokens)
        if return_all_layers:
            all_layers[-1] = tokens
            return tokens, (h, w), all_layers
        return tokens, (h, w)


class BandAdapter(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / in_chans)
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj(x)


def attach_band_adapter(encoder: "ViTEncoder", checkpoint_band_count: int) -> None:
    old_proj = encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        checkpoint_band_count, old_proj.out_channels,
        kernel_size=old_proj.kernel_size, stride=old_proj.stride,
    )
    encoder.patch_embed.proj = new_proj
    encoder.patch_embed.band_adapter = BandAdapter(NUM_BANDS, checkpoint_band_count)


def load_external_checkpoint(
    encoder: ViTEncoder,
    checkpoint_path: str,
    checkpoint_state_key: Optional[str] = None,
) -> Dict[str, List[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt[checkpoint_state_key] if checkpoint_state_key else ckpt.get("model_state_dict", ckpt)

    own_state = encoder.state_dict()
    loaded, shape_mismatch, missing = [], [], []
    for name, param in state.items():
        if name not in own_state:
            missing.append(name)
            continue
        if own_state[name].shape != param.shape:
            shape_mismatch.append(f"{name}: ckpt{tuple(param.shape)} vs model{tuple(own_state[name].shape)}")
            continue
        own_state[name].copy_(param)
        loaded.append(name)

    encoder.load_state_dict(own_state)
    report = {
        "loaded": loaded,
        "skipped_shape_mismatch": shape_mismatch,
        "skipped_missing_in_checkpoint": missing,
    }
    print(f"[encoder] External checkpoint load: {len(loaded)} tensors loaded, "
          f"{len(shape_mismatch)} skipped (shape mismatch), "
          f"{len(missing)} skipped (not found in checkpoint).")
    if shape_mismatch:
        print("[encoder] Shape-mismatched tensors (left at random init):")
        for m in shape_mismatch[:10]:
            print(f"    {m}")
    return report


class SimpleFeaturePyramidNeck(nn.Module):
    def __init__(self, embed_dim: int, out_chans: Tuple[int, int, int, int] = (64, 128, 256, 512)):
        super().__init__()
        c2, c3, c4, c5 = out_chans
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(8, embed_dim // 2), nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
        )
        self.proj2 = nn.Conv2d(embed_dim // 4, c2, kernel_size=1)

        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.proj3 = nn.Conv2d(embed_dim // 2, c3, kernel_size=1)

        self.proj4 = nn.Conv2d(embed_dim, c4, kernel_size=1)

        self.down2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.proj5 = nn.Conv2d(embed_dim, c5, kernel_size=1)

        self.out_chans = out_chans

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> List[torch.Tensor]:
        b, n, d = tokens.shape
        h, w = grid_hw
        grid = tokens.transpose(1, 2).reshape(b, d, h, w)  # (B, D, H', W')

        f2 = self.proj2(self.up4(grid))
        f3 = self.proj3(self.up2(grid))
        f4 = self.proj4(grid)
        f5 = self.proj5(self.down2(grid))
        return [f2, f3, f4, f5]  # strides: p/4, p/2, p, 2p (relative to input image)


def build_encoder(cfg: dict) -> ViTEncoder:
    return ViTEncoder(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=NUM_BANDS,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
    )

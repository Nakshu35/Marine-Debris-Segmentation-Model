from typing import Tuple

import torch
import torch.nn as nn

from data.transforms import NUM_BANDS
from models.encoder import TransformerBlock, ViTEncoder, sincos_pos_embed


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (B, N, D). Returns (x_visible, mask, ids_restore).
    mask: (B, N) with 1 = masked (removed), 0 = kept (visible).
    ids_restore: (B, N) indices to undo the shuffle later.
    """
    b, n, d = x.shape
    len_keep = int(n * (1 - mask_ratio))

    noise = torch.rand(b, n, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, d))

    mask = torch.ones(b, n, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)

    return x_visible, mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        patch_size: int,
        num_bands: int,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio=4.0
            )
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        self.pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * num_bands
        )

        self.decoder_embed_dim = decoder_embed_dim

    def forward(
        self,
        x_visible: torch.Tensor,
        ids_restore: torch.Tensor
    ) -> torch.Tensor:

        x = self.decoder_embed(x_visible)

        b, len_keep, dd = x.shape
        n = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(
            b,
            n - len_keep,
            1
        )

        x_full = torch.cat(
            [x, mask_tokens],
            dim=1
        )

        x_full = torch.gather(
            x_full,
            1,
            ids_restore.unsqueeze(-1).repeat(1, 1, dd)
        )

        # Create positional embedding using the ACTUAL token count
        grid_size = int(n ** 0.5)

        if grid_size * grid_size != n:
            raise ValueError(
                f"Number of tokens ({n}) is not a perfect square. "
                f"Cannot create 2D positional embedding."
            )

        pos_embed = sincos_pos_embed(
            self.decoder_embed_dim,
            grid_size
        )

        pos_embed = pos_embed.to(
            device=x_full.device,
            dtype=x_full.dtype
        )

        x_full = x_full + pos_embed

        for blk in self.blocks:
            x_full = blk(x_full)

        x_full = self.norm(x_full)

        return self.pred(x_full)


class MaskedAutoencoderSSL(nn.Module):
    def __init__(
        self,
        encoder: ViTEncoder,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = encoder.patch_size
        # Always NUM_BANDS: the raw input to the encoder is always the
        # 11-band Sentinel-2 stack, even when a BandAdapter internally
        # remaps it to an external checkpoint's native channel count
        # before its patch-embedding conv (see models/encoder.py).
        self.num_bands = NUM_BANDS

        self.decoder = MAEDecoder(
            encoder_embed_dim=encoder.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_size=self.patch_size,
            num_bands=self.num_bands,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, gh, gw, p, p, C)
        x = x.reshape(b, gh * gw, p * p * c)
        return x

    def forward_encoder_masked(self, imgs: torch.Tensor):
        tokens, (h, w) = self.encoder.patch_embed(imgs)
        tokens = tokens + self.encoder._pos_embed_for(h)
        x_visible, mask, ids_restore = random_masking(tokens, self.mask_ratio)
        for blk in self.encoder.blocks:
            x_visible = blk(x_visible)
        x_visible = self.encoder.norm(x_visible)
        return x_visible, mask, ids_restore

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (loss, pred, mask). mask: (B,N) 1=masked/evaluated."""
        x_visible, mask, ids_restore = self.forward_encoder_masked(imgs)
        pred = self.decoder(x_visible, ids_restore)  # (B, N, p*p*C)
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, pred, mask

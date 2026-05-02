"""
U-Net architecture for noise prediction in DDPM.

The network takes a noisy image and a timestep, then predicts the
noise that was added. It uses an encoder-decoder structure with
skip connections and sinusoidal time conditioning.

Architecture:
    Encoder: 64 -> 128 -> 256 -> 512 -> 1024
    Decoder: 1024 -> 512 -> 256 -> 128 -> 64
    Skip connections between corresponding encoder/decoder levels.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings.

    Encodes integer timesteps into continuous vectors using sinusoidal
    functions, analogous to positional encoding in Transformers
    (Vaswani et al., 2017).

    Args:
        dim: Dimension of the embedding vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """U-Net building block with time conditioning.

    Two convolutions with BatchNorm and ReLU, plus time embedding
    injection. Performs downsampling (strided conv) or upsampling
    (transposed conv) depending on the `up` flag.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        time_emb_dim: Dimension of the time embedding.
        up: If True, upsample (decoder). If False, downsample (encoder).
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    """U-Net for noise prediction in DDPM.

    Convolutional encoder-decoder with skip connections and timestep
    conditioning. The encoder progressively downsamples while increasing
    channel depth; the decoder reverses this using transposed convolutions
    and skip connections from corresponding encoder levels.

    Args:
        image_channels: Input image channels (default: 3 for RGB).
        down_channels: Channel sizes for the encoder path.
        up_channels: Channel sizes for the decoder path.
        time_emb_dim: Dimension of the sinusoidal time embedding.
    """

    def __init__(
        self,
        image_channels: int = 3,
        down_channels: tuple = (64, 128, 256, 512, 1024),
        up_channels: tuple = (1024, 512, 256, 128, 64),
        time_emb_dim: int = 32,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Predict the noise in a noisy image at a given timestep.

        Args:
            x: Noisy image tensor of shape (B, C, H, W).
            timestep: Integer timesteps of shape (B,).

        Returns:
            Predicted noise tensor of shape (B, C, H, W).
        """
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

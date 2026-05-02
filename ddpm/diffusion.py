"""
Diffusion process for DDPM.

Implements the forward (noise addition) and reverse (denoising) processes
of Denoising Diffusion Probabilistic Models with a linear beta schedule.

Reference:
    Ho, J., Jain, A., & Abbeel, P. (2020).
    "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def linear_beta_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02) -> torch.Tensor:
    """Linear schedule for noise variance (beta).

    Args:
        timesteps: Total number of diffusion steps.
        start: Starting beta value.
        end: Ending beta value.

    Returns:
        Tensor of shape (timesteps,) with linearly spaced betas.
    """
    return torch.linspace(start, end, timesteps)


def _extract(vals: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract values from a 1-D tensor at indices t and reshape for broadcasting.

    Args:
        vals: 1-D tensor of precomputed values.
        t: Batch of integer timestep indices, shape (B,).
        x_shape: Shape of the image tensor for broadcasting.

    Returns:
        Extracted values reshaped for broadcasting with images.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffusion:
    """DDPM diffusion process manager.

    Precomputes all noise schedule constants and provides methods
    for the forward process (adding noise) and reverse process
    (iterative denoising / sampling).

    Args:
        timesteps: Number of diffusion steps (default: 300).
        device: Torch device for computations.
    """

    def __init__(self, timesteps: int = 300, device: str = "cuda"):
        self.timesteps = timesteps
        self.device = device

        # Precompute schedule constants
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def forward_process(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean images at timestep t (forward / q-sample).

        Args:
            x_0: Clean images, shape (B, C, H, W).
            t: Timestep indices, shape (B,).
            noise: Gaussian noise, same shape as x_0.

        Returns:
            Noisy images at timestep t.
        """
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int = 16,
        channels: int = 3,
    ) -> torch.Tensor:
        """Generate images by iteratively denoising pure Gaussian noise.

        Args:
            model: Trained noise-prediction network (U-Net).
            image_size: Spatial resolution of generated images.
            batch_size: Number of images to generate.
            channels: Number of image channels.

        Returns:
            Generated images, shape (batch_size, channels, image_size, image_size).
        """
        img = torch.randn(
            (batch_size, channels, image_size, image_size), device=self.device
        )

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(img, t)

            alpha_t = _extract(self.alphas, t, img.shape)
            alpha_t_cumprod = _extract(self.alphas_cumprod, t, img.shape)
            beta_t = _extract(self.betas, t, img.shape)

            noise = torch.randn_like(img) if i > 1 else torch.zeros_like(img)

            model_mean = (1 / torch.sqrt(alpha_t)) * (
                img - (beta_t / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
            )
            posterior_var_t = _extract(self.posterior_variance, t, img.shape)
            img = model_mean + torch.sqrt(posterior_var_t) * noise

        return img

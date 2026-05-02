"""
DDPM — Denoising Diffusion Probabilistic Models for CIFAR-10.

Reference:
    Ho, J., Jain, A., & Abbeel, P. (2020).
    "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
"""

from .unet import UNet
from .diffusion import Diffusion

__version__ = "1.0.0"
__author__ = "Gabriel Yogi"
__all__ = ["UNet", "Diffusion"]

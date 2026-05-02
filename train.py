#!/usr/bin/env python3
"""
Train a DDPM (Denoising Diffusion Probabilistic Model) on CIFAR-10.

The model learns to predict the noise added at each diffusion timestep,
enabling high-quality image generation after training.

Usage:
    python train.py                          # defaults: 100 epochs, bs=128
    python train.py --epochs 50 --lr 2e-4    # custom hyperparameters
    python train.py --resume checkpoints/ddpm_epoch_50.pt  # resume training
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ddpm import UNet, Diffusion


def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--timesteps", type=int, default=300, help="Number of diffusion steps")
    p.add_argument("--sample_every", type=int, default=10, help="Save samples every N epochs")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--cpu", action="store_true", help="Force CPU training")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"🖥️  Device: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model & Diffusion
    model = UNet().to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=str(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"🔁 Resumed from {args.resume} (epoch {start_epoch})")
        else:
            model.load_state_dict(ckpt)
            print(f"🔁 Loaded weights from {args.resume}")

    # Output directories
    os.makedirs("ddpm_samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Training
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🧠 Model parameters: {param_count:.1f}M")
    print(f"📊 Dataset: {len(dataset):,} images | Batch: {args.batch_size}")
    print(f"🚀 Training for epochs {start_epoch}–{args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for images, _ in pbar:
            optimizer.zero_grad()

            images = images.to(device)
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device).long()
            noise = torch.randn_like(images)

            noisy_images = diffusion.forward_process(images, t, noise)
            predicted_noise = model(noisy_images, t)

            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save samples and checkpoint periodically
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                sample_imgs = diffusion.sample(model, image_size=32, batch_size=64)
                save_image(
                    sample_imgs,
                    f"ddpm_samples/epoch_{epoch:03d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                    nrow=8,
                )

            # Save full checkpoint (resumable)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoints/ddpm_epoch_{epoch}.pt",
            )
            print(f"  💾 Saved checkpoint & samples (epoch {epoch})")

    # Save final weights
    torch.save(model.state_dict(), "checkpoints/ddpm_final.pt")
    print(f"\n✅ Training complete! Final weights → checkpoints/ddpm_final.pt")


if __name__ == "__main__":
    main()

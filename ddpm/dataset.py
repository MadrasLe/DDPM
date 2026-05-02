"""
Rotation prediction dataset for self-supervised evaluation.

Implements RotNet (Gidaris et al., 2018) — a self-supervised pretext
task where the model learns visual features by predicting the rotation
angle applied to each image (0°, 90°, 180°, 270°).
"""

import os
import glob
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class RotationDataset(Dataset):
    """Dataset that applies random rotations and returns rotation labels.

    For each image, a random rotation from {0, 90, 180, 270} degrees is
    applied. The model must predict which rotation was used (4-class
    classification), learning useful visual features in the process.

    Args:
        root_dir: Directory containing .png images.
        transform: Optional torchvision transforms to apply after rotation.

    Raises:
        RuntimeError: If no .png images are found in root_dir.
    """

    ROTATIONS = [0, 90, 180, 270]

    def __init__(self, root_dir: str, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.transform = transform
        if not self.image_paths:
            raise RuntimeError(
                f"No .png images found in '{root_dir}'. "
                "Did you generate the synthetic dataset with the DDPM first?"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        rotation_idx = random.randint(0, 3)
        angle = self.ROTATIONS[rotation_idx]
        rotated_image = transforms.functional.rotate(image, angle)

        if self.transform:
            rotated_image = self.transform(rotated_image)

        return rotated_image, rotation_idx

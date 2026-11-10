"""Dataset y transformaciones en PyTorch para Colorectal Histology.

Gestion de memoria eficiente:
- Mantiene los arrays base en memoria en uint8 [0, 255].
- Convierte a tensores float32 (C, H, W) normalizados en [0.0, 1.0] sobre la marcha por lote.
"""

from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ColorectalDataset(Dataset):
    """Dataset de teselas histologicas colorrectales (150x150x3)."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            images: Array uint8 de forma (N, 150, 150, 3).
            labels: Array int64 de forma (N,).
            transform: Transformacion opcional aplicada a cada imagen (PIL/Tensor).
        """
        assert len(images) == len(labels), "images y labels deben tener la misma longitud"
        self.images = images
        self.labels = labels
        self.transform = transform

        # Transformacion por defecto si no se pasa ninguna: uint8 -> float32 [0.0, 1.0]
        self.default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img_arr = self.images[idx] # (150, 150, 3), uint8
        label = int(self.labels[idx])

        if self.transform is not None:
            # Convertir a PIL Image para permitir transformaciones estandar de torchvision
            img_pil = Image.fromarray(img_arr)
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = self.default_to_tensor(img_arr)

        return img_tensor, torch.tensor(label, dtype=torch.long)


def get_lenet_transforms(img_size: int = 32):
    """Transformaciones para el baseline LeNet-5 (redimensionado a 32x32 sin data augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), # -> float32 (3, 32, 32) en [0.0, 1.0]
    ])

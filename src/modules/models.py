"""Modelos convolucionales para el Desafio de Histologia Colorrectal.

Baseline LeNet-5 (LeCun et al. 1998 / Keras notebook del profesor):
- Redimensionado a 32x32.
- Activaciones Sigmoid en todas las capas ocultas.
- Optimizador SGD (lr=0.01), 25 epocas, batch 32.
- Manejo del desbalance mediante pesos inversos de clase en CrossEntropyLoss.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ColorectalDataset, get_lenet_transforms


class LeNet5(nn.Module):
    """Topologia exacta de LeNet-5 adaptada a 3 canales RGB (notebook del profesor)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # Bloque convolucional 1: (3, 32, 32) -> (6, 28, 28) -> (6, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

        # Capas densas: (16 * 5 * 5 = 400) -> 120 -> 84 -> num_classes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque 1
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        # Bloque 2
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        # Aplanado y capas densas
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)  # Devolvemos logits directamente
        return x


class LeNet5BaselineClassifier:
    """Clasificador envoltorio estilo scikit-learn/NLP para LeNet-5 baseline."""

    def __init__(
        self,
        lr: float = 0.01,
        epochs: int = 25,
        batch_size: int = 32,
        seed: int = 42,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[LeNet5] = None
        self.transform = get_lenet_transforms(img_size=32)

    def fit(self, train_images: np.ndarray, train_labels: np.ndarray) -> "LeNet5BaselineClassifier":
        """Entrena LeNet-5 sobre el conjunto de entrenamiento de un pliegue."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = LeNet5(in_channels=3, num_classes=2).to(self.device)

        # Calcular pesos balanceados estrictamente sobre el pliegue de train (cero fuga)
        classes = np.unique(train_labels)
        weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        train_dataset = ColorectalDataset(train_images, train_labels, transform=self.transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            for batch_imgs, batch_y in train_loader:
                batch_imgs = batch_imgs.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_imgs)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_imgs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            if self.verbose and (epoch % 5 == 0 or epoch == self.epochs):
                print(f"    Epoca {epoch:2d}/{self.epochs} - Loss: {epoch_loss:.4f}")

        return self

    def predict_proba(self, test_images: np.ndarray) -> np.ndarray:
        """Devuelve la probabilidad continua predicha para la clase positiva (1: TUMOR)."""
        assert self.model is not None, "El modelo debe ser entrenado antes de predecir"
        self.model.eval()

        test_dataset = ColorectalDataset(test_images, np.zeros(len(test_images), dtype=int), transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch_imgs, _ in test_loader:
                batch_imgs = batch_imgs.to(self.device)
                logits = self.model(batch_imgs)
                # Softmax para obtener probabilidades continuas reales (sin redondear)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list)

    def predict(self, test_images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Devuelve las clases predichas con umbral de decision."""
        probs = self.predict_proba(test_images)
        return (probs >= threshold).astype(int)

"""
Fase 0 — LeNet-5 sobre FashionMNIST (32x32) con PyTorch.

Réplica del baseline de Keras del profesor:
  - imágenes redimensionadas a 32x32 (equivalente a tf.image.resize)
  - activación sigmoid
  - max pooling
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SEMILLA = 0
BATCH_SIZE = 32      
N_EPOCAS = 25        

torch.manual_seed(SEMILLA)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Dispositivo: {device}")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
transformacion = transforms.Compose([
    transforms.Resize((32, 32)),   # 28x28 -> 32x32, como tf.image.resize
    transforms.ToTensor(),         # -> float32 (1, 32, 32) en [0, 1]
])

train_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transformacion)
test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=transformacion)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# Modelo — TODO: escribe tú esta clase
# ---------------------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # la capa de entrada nunca se crea, es x de forward
        ###### En __init__ se crea; en forward se aplica. #######
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # se reusan
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):                        # x: (N, 1, 32, 32)
        x = self.conv1(x)                        # (N, 6, 28, 28)
        x = self.sigmoid(x)                      # (N, 6, 28, 28)
        x = self.pool(x)                         # (N, 6, 14, 14)

        x = self.conv2(x)                        # (N, 16, 10, 10)
        x = self.sigmoid(x)                      # (N, 16, 10, 10)
        x = self.pool(x)                         # (N, 16, 5, 5)

        x = torch.flatten(x, start_dim=1)        # (N, 400)

        x = self.fc1(x)                          # (N, 120)
        x = self.sigmoid(x)
        x = self.fc2(x)                          # (N, 84)
        x = self.sigmoid(x)
        x = self.fc3(x)                          # (N, 10), sin activación

        return x


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def entrenar_una_epoca(modelo, loader, criterio, optimizador):
    modelo.train()
    loss_total = 0.0
    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizador.zero_grad()
        prediccion = modelo(imagenes)
        loss = criterio(prediccion, etiquetas)
        loss.backward()
        optimizador.step()

        loss_total += loss.item() * imagenes.size(0)   # el último batch es más pequeño
    return loss_total / len(loader.dataset)


def evaluar(modelo, loader):
    modelo.eval()
    aciertos = 0
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)
            clase_predicha = modelo(imagenes).argmax(dim=1)
            aciertos += (clase_predicha == etiquetas).sum().item()
    return aciertos / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    modelo = LeNet5().to(device)

    n_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros: {n_parametros}  (esperado: 61706)")

    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01)

    for epoca in range(1, N_EPOCAS + 1):
        inicio = time.perf_counter()
        loss_train = entrenar_una_epoca(modelo, train_loader, criterio, optimizador)
        acc_test = evaluar(modelo, test_loader)
        duracion = time.perf_counter() - inicio
        print(f"Época {epoca:2d} | loss train {loss_train:.4f} | "
              f"acc test {acc_test:.4f} | {duracion:.1f} s")

    torch.save(modelo.state_dict(), "lenet5_fashionmnist.pt")
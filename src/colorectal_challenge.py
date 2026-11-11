"""
Desafio de Histologia Colorrectal (CNN en PyTorch)
===================================================

Contenido del Nivel Basico:
1. Carga y gestion eficiente de memoria del dataset (Zenodo 53169, 5.000 teselas).
2. Particiones fijas de validacion cruzada estratificada de 10 pliegues (10-CV).
3. Baseline LeNet-5 (LeCun 1998 / Keras notebook del profesor):
   - Resolucion 32x32
   - Activaciones Sigmoid
   - Optimizador SGD (lr=0.01)
   - Compensacion del desbalance 1:7 mediante perdida ponderada (class weights)
   - Evaluacion rigurosa en ROC-AUC continua (sin redondeo artificial)
4. Metricas y evaluacion Out-Of-Fold (OOF).
"""

from pathlib import Path
from typing import Callable, Optional, Type
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1. CONSTANTES Y CONFIGURACION DEL ENTORNO
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 10

# Rutas del proyecto relativas al script
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Seleccion de dispositivo jerarquica: Apple Silicon (MPS) -> NVIDIA (CUDA) -> CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# 2. CARGA DE DATOS Y GESTION DE MEMORIA
# ---------------------------------------------------------------------------

# Carga las imagenes y etiquetas preprocesadas en uint8 y las particiones 10-CV
def load_dataset() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Gestion de memoria:
    - Las imagenes originales (5.000 x 150 x 150 x 3) se almacenan en uint8 (~322 MB).
    - Esto evita saturar los 8 GB de RAM del MacBook M2, ya que en float32 ocuparian
      1,35 GB solo en memoria cruda sin contar tensores intermedios ni gradientes.
    
    Returns:
        images: Array numpy uint8 con las 5.000 teselas.
        labels: Array numpy int64 (1: TUMOR, 0: NO TUMOR).
        splits: Lista con los 10 pliegues estratificados.
    """
    img_path = DATA_DIR / "images_150x150.npy"
    lbl_path = DATA_DIR / "labels_binary.npy"
    splits_path = DATA_DIR / "splits_10cv.json"

    if not img_path.exists() or not lbl_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            "No se encontraron los datos procesados en data/. "
            "Asegurate de que existen images_150x150.npy, labels_binary.npy y splits_10cv.json."
        )

    images = np.load(img_path)
    labels = np.load(lbl_path)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Usamos la estratificacion multiclase: garantiza que en cada pliegue
    # la proporcion de TUMOR vs NO TUMOR sea 1:7 y que los 7 tipos de tejido
    # sano se repartan equitativamente entre los 10 pliegues.
    splits = splits_data["splits_multiclass_stratified"]

    return images, labels, splits


# Dataset de PyTorch para teselas histologicas
class ColorectalDataset(Dataset):
    """
    Aplica transformaciones de imagen sobre la marcha por cada lote,
    garantizando que la memoria de la GPU solo contenga un lote float32 a la vez.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]  # uint8 en rango [0, 255]
        label = int(self.labels[idx])

        if self.transform is not None:
            # Si hay transformacion (por ejemplo, Resize o Data Augmentation),
            # convertimos temporalmente a PIL Image
            img_pil = Image.fromarray(img_arr)
            img_tensor = self.transform(img_pil)
        else:
            # Conversion estandar: uint8 [0, 255] -> float32 [0.0, 1.0] con forma (3, H, W)
            img_tensor = self.default_to_tensor(img_arr)

        return img_tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# 3. BASELINE: ARQUITECTURA LeNet-5
# ---------------------------------------------------------------------------

# LeNet-5 original adaptada a 3 canales RGB 
class LeNet5(nn.Module):
    """
    Estructura capa a capa:
    1. Conv2D(in=3, out=6, kernel=5x5) -> Sigmoid -> MaxPool(2x2, stride=2)
       Entrada: (3, 32, 32) -> Salida: (6, 14, 14)
    2. Conv2D(in=6, out=16, kernel=5x5) -> Sigmoid -> MaxPool(2x2, stride=2)
       Entrada: (6, 14, 14) -> Salida: (16, 5, 5)
    3. Flatten: 16 * 5 * 5 = 400 neuronas
    4. Densa(400 -> 120) -> Sigmoid
    5. Densa(120 -> 84) -> Sigmoid
    6. Densa(84 -> 2) -> Logits finales (sin softmax, se integra en CrossEntropyLoss)
    
    Total parametros: 61.326 pesos entrenables.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque convolucional 1
        #   1. conv1 aplica 6 filtros para buscar caracteristicas basicas
        #   2. sigmoid activa la neurona de forma no lineal, los valores quedan entre 0 y 1
        #   3. pool reduce el tamaño de la imagen a la mitad,quedandose con lo mas importante
        x = self.pool(self.sigmoid(self.conv1(x)))
        # Bloque convolucional 2
        # Lo mismo pero con 16 filtros
        x = self.pool(self.sigmoid(self.conv2(x)))

        # Aplanado para entrar en las capas densas (1 fila de numeros)
        x = torch.flatten(x, start_dim=1)

        # Capas densas conectadas
        # Pasan la fila por nuuronas conectadas entre sí en las capas ocultas
        # sigmoid mantiene las activaciones controladas
        x = self.sigmoid(self.fc1(x)) 
        x = self.sigmoid(self.fc2(x))
        # Transforma los datos en la "puntuacion final"
        # Los logits son las puntuaciones de cada clase, la mas alta es la que se decide como prediccion
        x = self.fc3(x)  # Logits de salida
        return x


# -------------------------------------------------------------
# MEJORA 1: LeNet-5 con activaciones ReLU
# -------------------------------------------------------------
# Misma arquitectura y parametros (61.326 pesos), pero sustituyendo las sigmoides por ReLU para resolver el desvanecimiento del gradiente
class LeNet5ReLU(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU: derivada = 1 para valores > 0, el gradiente no se reduce a 0.25
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque convolucional 1 con ReLU
        x = self.pool(self.relu(self.conv1(x)))
        # Bloque convolucional 2 con ReLU
        x = self.pool(self.relu(self.conv2(x)))

        # Aplanado a vector 1D
        x = torch.flatten(x, start_dim=1)

        # Capas densas con ReLU
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Logits de salida
        return x


# ---------------------------------------------------------------------------
# 3. ENVOLTORIO UNIFICADO Y CONFIGURABLE (Patron Estrategia)
# ---------------------------------------------------------------------------
class HistologyClassifier:
    """
    Envoltorio unificado para entrenamiento e inferencia de CNNs histologicas.
    
    Aplica el patron Estrategia para evitar duplicar codigo entre experimentos:
    permite desacoplar la arquitectura (LeNet-5 Sigmoid, LeNet-5 ReLU, etc.)
    del optimizador (SGD, Adam, AdamW), garantizando que todos los modelos se
    evaluan bajo exactamente el mismo protocolo de cero fuga y gestion de memoria.
    """

    def __init__(
        self,
        model_cls: Type[nn.Module] = LeNet5ReLU,
        optimizer_name: str = "sgd",
        lr: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 25,
        batch_size: int = 32,
        seed: int = SEED,
        device: torch.device = DEVICE,
    ):
        self.model_cls = model_cls
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.model: Optional[nn.Module] = None

    def fit(self, train_images: np.ndarray, train_labels: np.ndarray) -> "HistologyClassifier":
        """
        Entrena el modelo en el conjunto de entrenamiento de UN pliegue.
        
        Aislamiento y cero fuga:
        - Fijamos la semilla en cada pliegue para asegurar reproducibilidad.
        - Los pesos de compensacion del desbalance se calculan SOLO con train_labels.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # 1. Instanciar la red neuronal limpia para este pliegue
        self.model = self.model_cls(in_channels=3, num_classes=2).to(self.device)

        # 2. Pesos balanceados (inversos a la frecuencia) calculados solo con train
        #    Compensa el desbalance 1:7 entre TUMOR (625) y NO TUMOR (4.375)
        classes = np.unique(train_labels)
        weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # 3. Configuracion del optimizador segun el experimento
        if self.optimizer_name == "sgd":
            # Descenso por Gradiente Estocastico clasico
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            # Momentos adaptativos (media e inercia del gradiente por parametro)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            # Momentos adaptativos con decaimiento de pesos desacoplado (regularizacion limpia)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Optimizador no soportado: {self.optimizer_name}")

        # 4. DataLoader para procesar lote a lote sin saturar la memoria GPU/MPS
        train_dataset = ColorectalDataset(train_images, train_labels, transform=None)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # 5. Bucle de entrenamiento por epocas
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for batch_imgs, batch_y in train_loader:
                batch_imgs = batch_imgs.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_imgs)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, test_images: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad continua de la clase positiva (TUMOR, clase 1)
        aplicando la funcion Softmax real, sin ningun redondeo.
        """
        assert self.model is not None, "El modelo debe estar entrenado antes de predecir."
        self.model.eval()

        test_dataset = ColorectalDataset(test_images, np.zeros(len(test_images), dtype=int), transform=None)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch_imgs, _ in test_loader:
                batch_imgs = batch_imgs.to(self.device)
                logits = self.model(batch_imgs)
                # Softmax continuo: seleccionamos la columna 1 (probabilidad de TUMOR)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list)

    def predict(self, test_images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Prediccion discreta (0 o 1) aplicando el umbral clinico estandar (0.5).
        """
        probs = self.predict_proba(test_images)
        return (probs >= threshold).astype(int)


# Fabricas de conveniencia para invocar cada configuracion experimental
def BaselineLeNet5Classifier(lr=0.01, epochs=25, batch_size=32, seed=SEED, device=DEVICE):
    """Modelo Base: LeNet-5 original con Sigmoid y SGD (lr=0.01)"""
    return HistologyClassifier(model_cls=LeNet5, optimizer_name="sgd", lr=lr, epochs=epochs, batch_size=batch_size, seed=seed, device=device)

def LeNet5ReLUClassifier(lr=0.01, epochs=25, batch_size=32, seed=SEED, device=DEVICE):
    """Mejora 1: LeNet-5 con ReLU y SGD (lr=0.01)"""
    return HistologyClassifier(model_cls=LeNet5ReLU, optimizer_name="sgd", lr=lr, epochs=epochs, batch_size=batch_size, seed=seed, device=device)

def LeNet5ReLUAdamClassifier(lr=0.001, epochs=25, batch_size=32, seed=SEED, device=DEVICE):
    """Experimento 2A: LeNet-5 con ReLU y Adam (lr=0.001)"""
    return HistologyClassifier(model_cls=LeNet5ReLU, optimizer_name="adam", lr=lr, epochs=epochs, batch_size=batch_size, seed=seed, device=device)

def LeNet5ReLUAdamWClassifier(lr=0.001, weight_decay=1e-4, epochs=25, batch_size=32, seed=SEED, device=DEVICE):
    """Experimento 2B: LeNet-5 con ReLU y AdamW (lr=0.001, weight_decay=1e-4)"""
    return HistologyClassifier(model_cls=LeNet5ReLU, optimizer_name="adamw", lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, seed=seed, device=device)






# ---------------------------------------------------------------------------
# 4. ARNES DE VALIDACION CRUZADA (10-CV) Y METRICAS
# ---------------------------------------------------------------------------

# Calcula las metricas requeridas: ROC-AUC continua, Accuracy y F1-Score
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = float(metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    acc = float(metrics.accuracy_score(y_true, y_pred))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "f1": f1,
    }


# Arnes de validacion cruzada de 10 pliegues (10-CV)
def evaluate_cv(
    model_name: str,
    model_factory: Callable,
    images: np.ndarray,
    labels: np.ndarray,
    splits: list[dict],
    return_oof: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    oof_rows = []

    print(f"\nIniciando evaluacion 10-CV: {model_name}")
    print("-" * 60)

    for fold_info in splits:
        k = fold_info["fold"]
        train_idx = np.array(fold_info["train_indices"])
        test_idx = np.array(fold_info["test_indices"])

        train_imgs, train_y = images[train_idx], labels[train_idx]
        test_imgs, test_y = images[test_idx], labels[test_idx]

        # 1. Crear modelo limpio para el pliegue k
        model = model_factory()

        # 2. Entrenar solo con datos de train
        t0 = time.perf_counter()
        model.fit(train_imgs, train_y)
        train_duration = time.perf_counter() - t0

        # 3. Evaluar en el conjunto de test
        y_prob = model.predict_proba(test_imgs)
        y_pred = model.predict(test_imgs)

        # 4. Registrar metricas
        fold_res = compute_metrics(test_y, y_prob, y_pred)
        row = {
            "model": model_name,
            "fold": k,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **fold_res,
        }
        rows.append(row)

        print(
            f"  Pliegue {k:2d} | ROC-AUC: {fold_res['roc_auc']:.4f} | "
            f"Accuracy: {fold_res['accuracy']:.4f} | F1: {fold_res['f1']:.4f} "
            f"({train_duration:.1f} s)"
        )

        # 5. Acumular predicciones OOF
        if return_oof:
            for i, idx_original in enumerate(test_idx):
                oof_rows.append({
                    "model": model_name,
                    "fold": k,
                    "sample_idx": int(idx_original),
                    "y_true": int(test_y[i]),
                    "y_prob": float(y_prob[i]),
                    "y_pred": int(y_pred[i]),
                })

    metrics_df = pd.DataFrame(rows)
    if return_oof:
        oof_df = pd.DataFrame(oof_rows)
        return metrics_df, oof_df
    return metrics_df


# Calcula la media y desviacion tipica entre los 10 pliegues
def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    return summary.round(4)


# Test de rangos con signo de Wilcoxon sobre pliegues emparejados
# Comprueba si la diferencia entre dos modelos en los 10 pliegues es estadisticamente significativa
def wilcoxon_compare(results: pd.DataFrame, model_a: str, model_b: str, metric: str = "roc_auc") -> dict:
    a = results[results["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = results[results["model"] == model_b].sort_values("fold")[metric].to_numpy()
    assert len(a) == len(b), "Los dos modelos deben haber sido evaluados sobre los mismos pliegues."

    stat, p_value = wilcoxon(a, b)
    return {
        "metric": metric,
        "a": model_a,
        "b": model_b,
        "median_diff_a_minus_b": float(np.median(a - b)),
        "folds_a_better": int(np.sum(a > b)),  # En ROC-AUC mayor es mejor
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# 5. EJECUCION PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("DESAFIO DE HISTOLOGIA COLORRECTAL — NIVEL BASICO (10-CV)")
    print(f"Dispositivo de computo: {DEVICE}")
    print("=" * 65)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    images_raw, labels_bin, splits = load_dataset()
    print(f"Dataset cargado: {len(images_raw)} imagenes (forma {images_raw.shape[1:]})")
    print(f"Distribucion de etiquetas: NO TUMOR={np.sum(labels_bin == 0)}, TUMOR={np.sum(labels_bin == 1)}")

    # 2. Pre-redimensionado a 32x32 para el baseline LeNet-5 (optimiza tiempo en RAM)
    print("\nPre-redimensionando teselas a 32x32 para LeNet-5...")
    t0 = time.perf_counter()
    images_32 = np.stack([np.array(Image.fromarray(im).resize((32, 32), Image.BILINEAR)) for im in images_raw])
    print(f"Redimensionado listo en {time.perf_counter() - t0:.2f} s (tamano en RAM: {images_32.nbytes / (1024*1024):.1f} MB)")

    # 3. Baseline LeNet-5 (Cargar si ya existe para no repetir computo, o ejecutar)
    baseline_cv_file = RESULTS_DIR / "baseline_lenet5_cv_results.csv"
    if baseline_cv_file.exists():
        print("\nCargando resultados previos del Baseline LeNet-5 desde disco...")
        baseline_metrics_df = pd.read_csv(baseline_cv_file)
    else:
        def create_baseline():
            return LeNet5BaselineClassifier(lr=0.01, epochs=25, batch_size=32, seed=SEED, device=DEVICE)

        baseline_metrics_df, baseline_oof_df = evaluate_cv(
            model_name="baseline_lenet5",
            model_factory=create_baseline,
            images=images_32,
            labels=labels_bin,
            splits=splits,
            return_oof=True,
        )
        baseline_metrics_df.to_csv(baseline_cv_file, index=False)
        baseline_oof_df.to_csv(RESULTS_DIR / "baseline_lenet5_oof.csv", index=False)
        summarize_results(baseline_metrics_df).to_csv(RESULTS_DIR / "baseline_lenet5_summary.csv")

    # 4. MEJORA 1: LeNet-5 con activaciones ReLU y SGD (Cargar si ya existe)
    relu_cv_file = RESULTS_DIR / "lenet5_relu_cv_results.csv"
    relu_oof_file = RESULTS_DIR / "lenet5_relu_oof.csv"
    relu_summary_file = RESULTS_DIR / "lenet5_relu_summary.csv"

    if relu_cv_file.exists() and relu_oof_file.exists():
        print("\nCargando resultados previos de LeNet-5 ReLU (SGD) desde disco...")
        relu_metrics_df = pd.read_csv(relu_cv_file)
        relu_oof_df = pd.read_csv(relu_oof_file)
    else:
        def create_relu():
            return LeNet5ReLUClassifier(lr=0.01, epochs=25, batch_size=32, seed=SEED, device=DEVICE)

        relu_metrics_df, relu_oof_df = evaluate_cv(
            model_name="lenet5_relu",
            model_factory=create_relu,
            images=images_32,
            labels=labels_bin,
            splits=splits,
            return_oof=True,
        )
        relu_metrics_df.to_csv(relu_cv_file, index=False)
        relu_oof_df.to_csv(relu_oof_file, index=False)
        summarize_results(relu_metrics_df).to_csv(relu_summary_file)

    # 5. EXPERIMENTO 2A: LeNet-5 ReLU con optimizador Adam (lr=0.001)
    adam_cv_file = RESULTS_DIR / "lenet5_relu_adam_cv_results.csv"
    adam_oof_file = RESULTS_DIR / "lenet5_relu_adam_oof.csv"
    adam_summary_file = RESULTS_DIR / "lenet5_relu_adam_summary.csv"

    if adam_cv_file.exists() and adam_oof_file.exists():
        print("\nCargando resultados previos de LeNet-5 ReLU (Adam) desde disco...")
        adam_metrics_df = pd.read_csv(adam_cv_file)
        adam_oof_df = pd.read_csv(adam_oof_file)
    else:
        def create_adam():
            return LeNet5ReLUAdamClassifier(lr=0.001, epochs=25, batch_size=32, seed=SEED, device=DEVICE)

        adam_metrics_df, adam_oof_df = evaluate_cv(
            model_name="lenet5_relu_adam",
            model_factory=create_adam,
            images=images_32,
            labels=labels_bin,
            splits=splits,
            return_oof=True,
        )
        adam_metrics_df.to_csv(adam_cv_file, index=False)
        adam_oof_df.to_csv(adam_oof_file, index=False)
        summarize_results(adam_metrics_df).to_csv(adam_summary_file)

    # 6. EXPERIMENTO 2B: LeNet-5 ReLU con optimizador AdamW (lr=0.001, weight_decay=1e-4)
    adamw_cv_file = RESULTS_DIR / "lenet5_relu_adamw_cv_results.csv"
    adamw_oof_file = RESULTS_DIR / "lenet5_relu_adamw_oof.csv"
    adamw_summary_file = RESULTS_DIR / "lenet5_relu_adamw_summary.csv"

    if adamw_cv_file.exists() and adamw_oof_file.exists():
        print("\nCargando resultados previos de LeNet-5 ReLU (AdamW) desde disco...")
        adamw_metrics_df = pd.read_csv(adamw_cv_file)
        adamw_oof_df = pd.read_csv(adamw_oof_file)
    else:
        def create_adamw():
            return LeNet5ReLUAdamWClassifier(lr=0.001, weight_decay=1e-4, epochs=25, batch_size=32, seed=SEED, device=DEVICE)

        adamw_metrics_df, adamw_oof_df = evaluate_cv(
            model_name="lenet5_relu_adamw",
            model_factory=create_adamw,
            images=images_32,
            labels=labels_bin,
            splits=splits,
            return_oof=True,
        )
        adamw_metrics_df.to_csv(adamw_cv_file, index=False)
        adamw_oof_df.to_csv(adamw_oof_file, index=False)
        summarize_results(adamw_metrics_df).to_csv(adamw_summary_file)

    # 7. TABLA COMPARATIVA CONSOLIDADA (10-CV)
    all_metrics = pd.concat([baseline_metrics_df, relu_metrics_df, adam_metrics_df, adamw_metrics_df], ignore_index=True)
    summary_all = summarize_results(all_metrics)

    print("\n" + "=" * 65)
    print("COMPARATIVA DE MODELOS 10-CV (Media ± Desv. Tipica):")
    print(summary_all.to_string())
    print("=" * 65)

    # 8. CONTRASTES ESTADISTICOS DE WILCOXON (N=10 pliegues)
    print("\n" + "=" * 65)
    print("TEST DE RANGOS CON SIGNO DE WILCOXON (Metrica: ROC-AUC, N=10)")
    print("=" * 65)

    comparisons = [
        ("lenet5_relu", "baseline_lenet5", "Mejora 1 (ReLU) vs. Baseline (Sigmoid)"),
        ("lenet5_relu_adam", "lenet5_relu", "Exp 2A (Adam) vs. Mejora 1 (SGD)"),
        ("lenet5_relu_adamw", "lenet5_relu_adam", "Exp 2B (AdamW) vs. Exp 2A (Adam)"),
        ("lenet5_relu_adamw", "baseline_lenet5", "Exp 2B (AdamW) vs. Baseline (Sigmoid)"),
    ]

    for mod_a, mod_b, desc in comparisons:
        w_res = wilcoxon_compare(all_metrics, mod_a, mod_b, metric="roc_auc")
        sig = "ESTADISTICAMENTE SIGNIFICATIVA (p < 0.05)" if w_res['p_value'] < 0.05 else "NO significativa (p >= 0.05)"
        print(f"\n* {desc}:")
        print(f"  Victorias de {mod_a}: {w_res['folds_a_better']} / 10 | Dif. medianas: {w_res['median_diff_a_minus_b']:+.4f} | p-valor: {w_res['p_value']:.5f}")
        print(f"  Veredicto: {sig}")

    print("\n" + "=" * 65)
    print("MATRICES DE CONFUSION GLOBALES OOF (5.000 instancias por modelo):")
    print("=" * 65)
    for name, df_oof in [("LeNet-5 ReLU (SGD)", relu_oof_df), ("LeNet-5 ReLU (Adam)", adam_oof_df), ("LeNet-5 ReLU (AdamW)", adamw_oof_df)]:
        cm = metrics.confusion_matrix(df_oof["y_true"], df_oof["y_pred"])
        print(f"\n{name}:")
        print("                     Pred: NO TUMOR (0)    Pred: TUMOR (1)")
        print(f"Real: NO TUMOR (0)          {cm[0, 0]:5d}               {cm[0, 1]:5d}")
        print(f"Real: TUMOR (1)             {cm[1, 0]:5d}               {cm[1, 1]:5d}")


if __name__ == "__main__":
    main()

"""
Desafio de Histologia Colorrectal (Nivel Basico y Medio en PyTorch)
===================================================================

Indice rapido del script (pensado para lectura directa):
  1. Configuracion, datos en uint8 y transformaciones (D4 y Lab-CLAHE)
  2. Arquitecturas CNN (LeNet5 y ModernColorectalCNN)
  3. Entrenador unificado (HistologyClassifier: optimizadores y CosineAnnealingLR)
  4. Validacion cruzada 10-CV, metricas y contrastes de Wilcoxon + Holm-Bonferroni
  5. Analisis de errores OOF y explicabilidad Grad-CAM (hooks en relu3)
  6. Ejecucion principal (Nivel Basico binario + Nivel Medio 8 clases)
"""

from pathlib import Path
from typing import Callable, Optional
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from sklearn import metrics
from sklearn.utils import class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1. CONFIGURACION, DATOS Y PREPROCESADO
# ---------------------------------------------------------------------------
SEED = 42
CLASS_NAMES = ["TUMOR", "STROMA", "COMPLEX", "LYMPHO", "DEBRIS", "MUCOSA", "ADIPOSE", "EMPTY"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# Seleccion automatica de aceleracion: Apple Silicon (MPS) -> NVIDIA (CUDA) -> CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(multiclass: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Carga imagenes en uint8 (~322 MB en RAM), etiquetas y particiones 10-CV fijas."""
    lbl_name = "labels_multiclass.npy" if multiclass else "labels_binary.npy"
    images = np.load(DATA_DIR / "images_150x150.npy")
    labels = np.load(DATA_DIR / lbl_name)
    with open(DATA_DIR / "splits_10cv.json", "r", encoding="utf-8") as f:
        splits = json.load(f)["splits_multiclass_stratified"]
    return images, labels, splits


class ColorectalDataset(Dataset):
    """Dataset PyTorch: convierte cada lote a float32 [0, 1] sobre la marcha para ahorrar RAM."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[Callable] = None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.images[idx]
        if self.transform is not None:
            img_tensor = self.transform(Image.fromarray(img_arr))
        else:
            img_tensor = self.to_tensor(img_arr)
        return img_tensor, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class RandomDiscreteRotation:
    """Rotacion ortogonal aleatoria (0, 90, 180 o 270 grados) sin bordes negros."""

    def __init__(self, angles: tuple[int, ...] = (0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(int(np.random.choice(self.angles)))


def get_d4_train_transform() -> transforms.Compose:
    """Aumento de datos biologico (grupo diedrico D4: 8 simetrias del portaobjetos)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomDiscreteRotation(),
        transforms.ToTensor(),
    ])


class ClaheLabTransform:
    """Aplica CLAHE solo al canal L de CIE Lab para preservar los tonos H&E (a, b)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        import cv2
        self._cv2 = cv2
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        lab = self._cv2.cvtColor(np.asarray(img, dtype=np.uint8), self._cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = self._cv2.split(lab)
        lab_eq = self._cv2.merge((self.clahe.apply(l_ch), a_ch, b_ch))
        return Image.fromarray(self._cv2.cvtColor(lab_eq, self._cv2.COLOR_LAB2RGB))


def get_clahe_lab_transform(clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> transforms.Compose:
    """Pipeline determinista imagen a imagen con Lab-CLAHE y paso a tensor."""
    return transforms.Compose([ClaheLabTransform(clip_limit, tile_grid_size), transforms.ToTensor()])


# ---------------------------------------------------------------------------
# 2. ARQUITECTURAS CNN EN PYTORCH
# Nota para lectura desde Keras: __init__ declara las capas y forward() define
# el flujo de tensores. Las redes devuelven logits sin Softmax porque
# nn.CrossEntropyLoss ya aplica LogSoftmax internamente de forma estable.
# ---------------------------------------------------------------------------
class LeNet5(nn.Module):
    """LeNet-5 (32x32 -> 61.326 parametros) con activacion configurable (sigmoid o relu)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, activation: str = "sigmoid"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU() if activation == "relu" else nn.Sigmoid()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class ModernColorectalCNN(nn.Module):
    """
    CNN moderna a 150x150 (Balaji et al., 2022): 3 bloques Conv-BN-ReLU-MaxPool + GAP + Dropout(0.2).
    Total: 25.842 parametros en binario (2 clases) y 26.040 en multiclase (8 clases).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # bias=False porque BatchNorm ya incorpora su propio parametro de desplazamiento (beta)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()  # Capa separada para registrar hooks de Grad-CAM sobre 37x37
        self.pool3 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling: evita millones de pesos
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(self.gap(x), start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 3. ENTRENADOR UNIFICADO (EQUIVALENTE A model.fit / model.predict DE KERAS)
# ---------------------------------------------------------------------------
class HistologyClassifier:
    """Envoltorio configurable para entrenar y predecir cualquier modelo bajo el mismo protocolo."""

    def __init__(
        self,
        model_factory_fn: Callable[[], nn.Module],
        num_classes: int = 2,
        optimizer_name: str = "adam",
        lr: float = 0.001,
        weight_decay: float = 0.0,
        epochs: int = 25,
        batch_size: int = 32,
        scheduler_name: Optional[str] = None,
        eta_min: float = 1e-5,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        seed: int = SEED,
        device: torch.device = DEVICE,
    ):
        self.model_factory_fn = model_factory_fn
        self.num_classes = num_classes
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.scheduler_name = scheduler_name.lower() if scheduler_name else None
        self.eta_min = eta_min
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.seed = seed
        self.device = device
        self.model: Optional[nn.Module] = None

    def fit(self, train_images: np.ndarray, train_labels: np.ndarray) -> "HistologyClassifier":
        """Entrena la red desde cero sobre un unico pliegue sin ver el conjunto de test."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = self.model_factory_fn().to(self.device)

        # En binario (1:7) compensamos el desbalance con pesos calculados solo sobre train
        if self.num_classes == 2:
            classes = np.unique(train_labels)
            weights = class_weight.compute_class_weight("balanced", classes=classes, y=train_labels)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        opt_map = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if self.optimizer_name not in opt_map:
            raise ValueError(f"Optimizador no soportado: {self.optimizer_name}")
        optimizer = opt_map[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = None
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.eta_min)

        loader = DataLoader(
            ColorectalDataset(train_images, train_labels, transform=self.train_transform),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()          # 1. Limpiar gradientes del paso anterior
                loss = criterion(self.model(batch_x), batch_y)  # 2. Forward + perdida
                loss.backward()                # 3. Backpropagation
                optimizer.step()               # 4. Actualizar pesos
            if scheduler is not None:
                scheduler.step()
        return self

    def predict_proba(self, test_images: np.ndarray) -> np.ndarray:
        """Devuelve probabilidades Softmax continuas: vector (N,) en binario o matriz (N, C) en multiclase."""
        assert self.model is not None, "Entrena el modelo antes de predecir."
        self.model.eval()
        loader = DataLoader(
            ColorectalDataset(test_images, np.zeros(len(test_images), dtype=int), transform=self.eval_transform),
            batch_size=self.batch_size,
            shuffle=False,
        )
        probs_list = []
        with torch.no_grad():  # Desactiva calculo de gradientes en inferencia (ahorra RAM y tiempo)
            for batch_x, _ in loader:
                probs = torch.softmax(self.model(batch_x.to(self.device)), dim=1)
                if self.num_classes == 2:
                    probs = probs[:, 1]
                probs_list.append(probs.cpu().numpy())
        return np.concatenate(probs_list)

    def predict(self, test_images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Prediccion discreta por umbral (binario) o argmax (multiclase)."""
        probs = self.predict_proba(test_images)
        return (probs >= threshold).astype(int) if self.num_classes == 2 else np.argmax(probs, axis=1).astype(int)


# ---------------------------------------------------------------------------
# 4. ARNES 10-CV, METRICAS Y CONTRASTES ESTADISTICOS
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula ROC-AUC (binario o macro OVR), Accuracy y F1 (binario o macro)."""
    is_mc = y_prob.ndim == 2 and y_prob.shape[1] > 2
    try:
        auc = float(metrics.roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") if is_mc else metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    return {
        "roc_auc": auc,
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "f1": float(metrics.f1_score(y_true, y_pred, average="macro" if is_mc else "binary", zero_division=0)),
    }


def evaluate_cv(
    model_name: str,
    model_factory: Callable[[], HistologyClassifier],
    images: np.ndarray,
    labels: np.ndarray,
    splits: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta validacion cruzada de 10 pliegues guardando metricas y predicciones OOF."""
    rows, oof_rows = [], []
    print(f"\nEvaluando 10-CV: {model_name}\n" + "-" * 55)

    for fold_info in splits:
        k = fold_info["fold"]
        train_idx, test_idx = np.array(fold_info["train_indices"]), np.array(fold_info["test_indices"])
        t0 = time.perf_counter()
        clf = model_factory().fit(images[train_idx], labels[train_idx])
        elapsed = time.perf_counter() - t0

        y_prob = clf.predict_proba(images[test_idx])
        y_pred = (y_prob >= 0.5).astype(int) if y_prob.ndim == 1 else np.argmax(y_prob, axis=1).astype(int)
        res = compute_metrics(labels[test_idx], y_prob, y_pred)
        rows.append({"model": model_name, "fold": k, "n_train": len(train_idx), "n_test": len(test_idx), **res})
        print(f"  Pliegue {k:2d} | AUC: {res['roc_auc']:.4f} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} ({elapsed:.1f} s)")

        is_mc = y_prob.ndim == 2 and y_prob.shape[1] > 2
        for i, orig_idx in enumerate(test_idx):
            entry = {"model": model_name, "fold": k, "sample_idx": int(orig_idx), "y_true": int(labels[orig_idx]), "y_pred": int(y_pred[i])}
            if is_mc:
                entry["confidence"] = float(np.max(y_prob[i]))
                for c in range(y_prob.shape[1]):
                    entry[f"prob_{c}"] = float(y_prob[i, c])
            else:
                entry["y_prob"] = float(y_prob[i])
            oof_rows.append(entry)

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Resume media y desviacion tipica por modelo sobre los 10 pliegues."""
    return results.groupby("model", sort=False).agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    ).round(4)


def wilcoxon_compare(results: pd.DataFrame, model_a: str, model_b: str, metric: str = "roc_auc") -> dict:
    """Test no parametrico de Wilcoxon pareado sobre los 10 pliegues."""
    a = results[results["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = results[results["model"] == model_b].sort_values("fold")[metric].to_numpy()
    _, p_value = wilcoxon(a, b)
    return {
        "median_diff": float(np.median(a - b)),
        "wins": int(np.sum(a > b)),
        "p_value": float(p_value),
    }


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Correccion secuencial de Holm-Bonferroni para comparaciones multiples."""
    m = len(p_values)
    adjusted = [1.0] * m
    prev_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(sorted(enumerate(p_values), key=lambda x: x[1])):
        prev_max = max(prev_max, min(1.0, p_val * (m - rank)))
        adjusted[orig_idx] = float(prev_max)
    return adjusted


def run_or_load_cv(
    model_name: str,
    model_factory: Callable[[], HistologyClassifier],
    images: np.ndarray,
    labels: np.ndarray,
    splits: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga resultados previos desde results/ o ejecuta la 10-CV y los guarda."""
    cv_file = RESULTS_DIR / f"{model_name}_cv_results.csv"
    oof_file = RESULTS_DIR / f"{model_name}_oof.csv"
    if cv_file.exists() and oof_file.exists():
        print(f"Cargando resultados previos de {model_name}...")
        return pd.read_csv(cv_file), pd.read_csv(oof_file)

    metrics_df, oof_df = evaluate_cv(model_name, model_factory, images, labels, splits)
    metrics_df.to_csv(cv_file, index=False)
    oof_df.to_csv(oof_file, index=False)
    summarize_results(metrics_df).to_csv(RESULTS_DIR / f"{model_name}_summary.csv")
    return metrics_df, oof_df


# ---------------------------------------------------------------------------
# 5. ANALISIS DE ERRORES OOF Y EXPLICABILIDAD (GRAD-CAM)
# ---------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM mediante hooks forward (mapas) y backward (gradientes) en relu3."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(lambda m, i, out: setattr(self, "activations", out.detach()))
        self._bwd = target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Devuelve el mapa de calor normalizado en [0, 1] reescalado a (150, 150)."""
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(next(self.model.parameters()).device).requires_grad_(True)
        self.model.zero_grad()
        self.model(x)[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=img_tensor.shape[1:], mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        return cam / float(cam.max()) if float(cam.max()) > 1e-8 else cam

    def remove_hooks(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def _save_gradcam_panels(cases: list[dict], out_path: Path, include_true_cam: bool, suptitle: str) -> None:
    """Dibuja y guarda en PNG la cuadricula de tesela original y superposiciones Grad-CAM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3 if include_true_cam else 2
    fig, axes = plt.subplots(len(cases), ncols, figsize=(4.2 * ncols, 3.8 * len(cases)))
    axes = np.expand_dims(axes, axis=0) if len(cases) == 1 else axes

    for row_idx, item in enumerate(cases):
        axes[row_idx, 0].imshow(item["image"])
        axes[row_idx, 0].set_title(f"#{item['sample_idx']} | Real: {item['true_name']}\nPred: {item['pred_name']} ({item['confidence']:.2%})", fontsize=9)
        axes[row_idx, 1].imshow(item["image"])
        axes[row_idx, 1].imshow(item["cam_pred"], cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"Grad-CAM clase predicha\n({item['pred_name']})", fontsize=9)
        if include_true_cam:
            axes[row_idx, 2].imshow(item["image"])
            axes[row_idx, 2].imshow(item["cam_true"], cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
            axes[row_idx, 2].set_title(f"Grad-CAM clase real\n({item['true_name']}, p={item['prob_true']:.2%})", fontsize=9)
        for col_idx in range(ncols):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_error_and_gradcam_analysis(images_raw: np.ndarray, labels_mc: np.ndarray, splits: list[dict], oof_df: pd.DataFrame) -> None:
    """Resume los errores OOF globales y genera mapas Grad-CAM sobre el pliegue 3."""
    filenames = np.load(DATA_DIR / "filenames.npy", allow_pickle=True)
    err_df = oof_df[oof_df["y_true"] != oof_df["y_pred"]].copy()
    err_df["true_name"] = err_df["y_true"].map(lambda i: CLASS_NAMES[int(i)])
    err_df["pred_name"] = err_df["y_pred"].map(lambda i: CLASS_NAMES[int(i)])
    print(f"\nANALISIS DE ERRORES OOF ({len(err_df)}/{len(oof_df)} fallos) — Top parejas:")
    for (t_name, p_name), cnt in err_df.groupby(["true_name", "pred_name"]).size().sort_values(ascending=False).head(8).items():
        print(f"  * {t_name} -> {p_name}: {cnt} casos")

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    cases_csv, err_png, ok_png = GRADCAM_DIR / "gradcam_cases.csv", GRADCAM_DIR / "gradcam_errors.png", GRADCAM_DIR / "gradcam_correct.png"
    if cases_csv.exists() and err_png.exists() and ok_png.exists():
        print("\nCargando casos Grad-CAM desde disco:\n" + pd.read_csv(cases_csv).to_string(index=False))
        return

    target_fold, err_ids, ok_ids = 3, [2738, 1847, 4317, 352], [420, 3663]
    clf = HistologyClassifier(lambda: ModernColorectalCNN(3, 8), num_classes=8, optimizer_name="adam", lr=0.001, scheduler_name="cosine")
    clf.fit(images_raw[splits[target_fold]["train_indices"]], labels_mc[splits[target_fold]["train_indices"]])

    grad_cam = GradCAM(clf.model, clf.model.relu3)
    to_tensor = transforms.ToTensor()
    error_cases, correct_cases, rows = [], [], []
    try:
        for idx in err_ids + ok_ids:
            img = images_raw[idx]
            probs = clf.predict_proba(np.expand_dims(img, axis=0))[0]
            true_c, pred_c = int(labels_mc[idx]), int(np.argmax(probs))
            entry = {
                "sample_idx": idx, "true_name": CLASS_NAMES[true_c], "pred_name": CLASS_NAMES[pred_c],
                "confidence": float(probs[pred_c]), "prob_true": float(probs[true_c]), "image": img,
                "cam_pred": grad_cam.generate(to_tensor(img), pred_c),
                "cam_true": grad_cam.generate(to_tensor(img), true_c),
            }
            rows.append({
                "kind": "error" if idx in err_ids else "correct", "sample_idx": idx, "fold": target_fold,
                "true_name": entry["true_name"], "pred_name": entry["pred_name"],
                "confidence": round(entry["confidence"], 4), "prob_true": round(entry["prob_true"], 4),
                "filename": str(filenames[idx]),
            })
            (error_cases if idx in err_ids else correct_cases).append(entry)
    finally:
        grad_cam.remove_hooks()

    _save_gradcam_panels(error_cases, err_png, True, "Grad-CAM (bloque conv3+BN+ReLU) sobre errores OOF del Pliegue 3 — mc_modern_cnn_150_cosine")
    _save_gradcam_panels(correct_cases, ok_png, False, "Grad-CAM (bloque conv3+BN+ReLU) sobre aciertos del Pliegue 3\nmc_modern_cnn_150_cosine")
    pd.DataFrame(rows).to_csv(cases_csv, index=False)


# ---------------------------------------------------------------------------
# 6. EJECUCION PRINCIPAL (NIVEL BASICO + NIVEL MEDIO)
# ---------------------------------------------------------------------------
def run_basic_level(images_raw: np.ndarray, splits: list[dict]) -> None:
    """Ejecuta o carga los 6 experimentos del Nivel Basico (binario 1:7) y sus contrastes."""
    _, labels_bin, _ = load_dataset(multiclass=False)
    lenet_names = ["baseline_lenet5", "lenet5_relu", "lenet5_relu_adam", "lenet5_relu_adamw"]
    images_32 = None
    if any(not (RESULTS_DIR / f"{name}_cv_results.csv").exists() for name in lenet_names):
        images_32 = np.stack([np.array(Image.fromarray(im).resize((32, 32), Image.BILINEAR)) for im in images_raw])

    basic_configs = [
        ("baseline_lenet5", images_32, lambda: HistologyClassifier(lambda: LeNet5(3, 2, "sigmoid"), 2, "sgd", lr=0.01)),
        ("lenet5_relu", images_32, lambda: HistologyClassifier(lambda: LeNet5(3, 2, "relu"), 2, "sgd", lr=0.01)),
        ("lenet5_relu_adam", images_32, lambda: HistologyClassifier(lambda: LeNet5(3, 2, "relu"), 2, "adam", lr=0.001)),
        ("lenet5_relu_adamw", images_32, lambda: HistologyClassifier(lambda: LeNet5(3, 2, "relu"), 2, "adamw", lr=0.001, weight_decay=1e-4)),
        ("modern_cnn_150", images_raw, lambda: HistologyClassifier(lambda: ModernColorectalCNN(3, 2), 2, "adam", lr=0.001)),
        ("modern_cnn_150_aug", images_raw, lambda: HistologyClassifier(lambda: ModernColorectalCNN(3, 2), 2, "adam", lr=0.001, train_transform=get_d4_train_transform())),
    ]

    results_map = {name: run_or_load_cv(name, factory, imgs, labels_bin, splits) for name, imgs, factory in basic_configs}
    all_metrics = pd.concat([m for m, _ in results_map.values()], ignore_index=True)

    print("\n" + "=" * 65 + "\nNIVEL BASICO — COMPARATIVA BINARIA 10-CV:\n" + summarize_results(all_metrics).to_string())
    comparisons = [
        ("lenet5_relu", "baseline_lenet5", "Mejora 1 (ReLU) vs Baseline"),
        ("lenet5_relu_adam", "lenet5_relu", "Exp 2A (Adam) vs Mejora 1 (SGD)"),
        ("lenet5_relu_adamw", "lenet5_relu_adam", "Exp 2B (AdamW) vs Exp 2A (Adam)"),
        ("modern_cnn_150", "lenet5_relu_adam", "Exp 3 (Modern CNN 150x150) vs LeNet-5 Adam"),
        ("modern_cnn_150_aug", "modern_cnn_150", "Exp 4 (Aug D4) vs Exp 3 (Sin Aug)"),
    ]
    print("\nTEST DE WILCOXON BINARIO (ROC-AUC, N=10):")
    for a, b, desc in comparisons:
        w = wilcoxon_compare(all_metrics, a, b, "roc_auc")
        print(f"  * {desc}: wins={w['wins']}/10 | dif_med={w['median_diff']:+.4f} | p={w['p_value']:.5f}")


def run_medium_level(images_raw: np.ndarray, splits: list[dict]) -> None:
    """Ejecuta o carga los 3 experimentos multiclase (8 clases), Wilcoxon+Holm y Grad-CAM."""
    _, labels_mc, _ = load_dataset(multiclass=True)
    clahe_tf = get_clahe_lab_transform(clip_limit=2.0, tile_grid_size=(8, 8))

    mc_configs = [
        ("mc_modern_cnn_150", lambda: HistologyClassifier(lambda: ModernColorectalCNN(3, 8), 8, "adam", lr=0.001)),
        ("mc_modern_cnn_150_clahe", lambda: HistologyClassifier(lambda: ModernColorectalCNN(3, 8), 8, "adam", lr=0.001, train_transform=clahe_tf, eval_transform=clahe_tf)),
        ("mc_modern_cnn_150_cosine", lambda: HistologyClassifier(lambda: ModernColorectalCNN(3, 8), 8, "adam", lr=0.001, scheduler_name="cosine", eta_min=1e-5)),
    ]

    results_map = {name: run_or_load_cv(name, factory, images_raw, labels_mc, splits) for name, factory in mc_configs}
    mc_metrics = pd.concat([m for m, _ in results_map.values()], ignore_index=True)

    print("\n" + "=" * 65 + "\nNIVEL MEDIO — COMPARATIVA MULTICLASE 8 CLASES 10-CV:\n" + summarize_results(mc_metrics).to_string())
    mc_pairs = [
        ("mc_modern_cnn_150_cosine", "mc_modern_cnn_150", "Exp M3 (Cosine) vs Exp M1 (lr fijo)"),
        ("mc_modern_cnn_150_cosine", "mc_modern_cnn_150_clahe", "Exp M3 (Cosine) vs Exp M2 (Lab-CLAHE)"),
        ("mc_modern_cnn_150_clahe", "mc_modern_cnn_150", "Exp M2 (Lab-CLAHE) vs Exp M1 (Cruda)"),
    ]
    print("\nTEST DE WILCOXON MULTICLASE CON AJUSTE HOLM-BONFERRONI (N=10):")
    for metric_name in ["roc_auc", "accuracy", "f1"]:
        raw = [wilcoxon_compare(mc_metrics, a, b, metric_name) for a, b, _ in mc_pairs]
        adj = holm_bonferroni_adjust([r["p_value"] for r in raw])
        print(f"  Metrica {metric_name.upper()}:")
        for (_, _, desc), w, p_h in zip(mc_pairs, raw, adj):
            print(f"    * {desc}: wins={w['wins']}/10 | dif_med={w['median_diff']:+.4f} | p={w['p_value']:.5f} | p_holm={p_h:.5f}")

    cosine_oof = results_map["mc_modern_cnn_150_cosine"][1]
    print("\nINFORME POR CLASE OOF — Campeon Multiclase (mc_modern_cnn_150_cosine):")
    print(metrics.classification_report(cosine_oof["y_true"], cosine_oof["y_pred"], target_names=CLASS_NAMES, digits=4, zero_division=0))
    run_error_and_gradcam_analysis(images_raw, labels_mc, splits, cosine_oof)


def main() -> None:
    print(f"DESAFIO DE HISTOLOGIA COLORRECTAL (10-CV) | Dispositivo: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images_raw, _, splits = load_dataset(multiclass=False)
    run_basic_level(images_raw, splits)
    run_medium_level(images_raw, splits)


if __name__ == "__main__":
    main()

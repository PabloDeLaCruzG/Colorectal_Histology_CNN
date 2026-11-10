"""Generacion y fijacion de particiones de validacion cruzada estratificada (10-CV).

Garantiza reproducibilidad estricta (SEED = 42) y cero fuga de datos.
Permite evaluar tanto estratificacion multiclase (preservando los 8 tejidos)
como estratificacion binaria directa (TUMOR vs NO TUMOR).
"""

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SEED = 42
N_SPLITS = 10


def generate_and_save_splits(seed: int = SEED, n_splits: int = N_SPLITS) -> dict:
    """Genera las particiones 10-CV estratificadas y las guarda en data/splits_10cv.json."""
    labels_multi_path = DATA_DIR / "labels_multiclass.npy"
    labels_bin_path = DATA_DIR / "labels_binary.npy"

    if not labels_multi_path.exists() or not labels_bin_path.exists():
        raise FileNotFoundError("Los ficheros de etiquetas no existen en data/. Ejecuta prepare_dataset.py primero.")

    y_multi = np.load(labels_multi_path)
    y_bin = np.load(labels_bin_path)

    # 1. Estratificacion sobre multiclase (0..7):
    # Al estratificar sobre las 8 clases originales, cada pliegue contiene exactamente
    # una proporcion uniforme de TUMOR (clase 0) y de cada uno de los otros 7 subtipos histologicos.
    skf_multi = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits_multi = []
    for fold, (train_idx, test_idx) in enumerate(skf_multi.split(np.zeros(len(y_multi)), y_multi)):
        splits_multi.append({
            "fold": fold,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_tumor_count": int(np.sum(y_bin[train_idx] == 1)),
            "test_tumor_count": int(np.sum(y_bin[test_idx] == 1)),
            "train_notumor_count": int(np.sum(y_bin[train_idx] == 0)),
            "test_notumor_count": int(np.sum(y_bin[test_idx] == 0)),
        })

    # 2. Estratificacion sobre binario directo (TUMOR vs NO TUMOR):
    skf_bin = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits_bin = []
    for fold, (train_idx, test_idx) in enumerate(skf_bin.split(np.zeros(len(y_bin)), y_bin)):
        splits_bin.append({
            "fold": fold,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_tumor_count": int(np.sum(y_bin[train_idx] == 1)),
            "test_tumor_count": int(np.sum(y_bin[test_idx] == 1)),
            "train_notumor_count": int(np.sum(y_bin[train_idx] == 0)),
            "test_notumor_count": int(np.sum(y_bin[test_idx] == 0)),
        })

    splits_data = {
        "seed": seed,
        "n_splits": n_splits,
        "splits_multiclass_stratified": splits_multi,
        "splits_binary_stratified": splits_bin,
    }

    out_file = DATA_DIR / "splits_10cv.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"Particiones 10-CV guardadas en {out_file}")
    return splits_data


if __name__ == "__main__":
    generate_and_save_splits()

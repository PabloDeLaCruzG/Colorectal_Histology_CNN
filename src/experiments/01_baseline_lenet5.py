"""Experimento 01: Evaluacion 10-CV del Baseline LeNet-5 (32x32, Sigmoid, SGD).

Replicacion fiel del punto de partida del profesor en tf.keras:
- Muestra el tiempo por pliegue medido en Apple Silicon (MPS).
- Guarda metricas de cada pliegue en results/baseline_lenet5_cv_results.csv.
- Guarda predicciones OOF (5000 muestras) en results/baseline_lenet5_oof.csv.
- Guarda resumen estadistico en results/baseline_lenet5_summary.csv.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics

from ..modules.evaluation import compute_metrics, evaluate_cv, summarize_results
from ..modules.models import LeNet5BaselineClassifier

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_lenet5_baseline(single_fold: bool = False) -> None:
    print(f"{'#' * 65}")
    print("EXPERIMENTO 01: BASELINE LeNet-5 (32x32, Sigmoid, SGD lr=0.01)")
    print(f"{'#' * 65}")

    # Cargar datos y pre-redimensionar a 32x32 una sola vez para acelerar 10-CV (14.6 MB en RAM)
    images_raw = np.load(DATA_DIR / "images_150x150.npy")
    labels_bin = np.load(DATA_DIR / "labels_binary.npy")

    print("Pre-redimensionando 5.000 teselas a 32x32 (14.6 MB en RAM)...")
    from PIL import Image
    images_32 = np.stack([np.array(Image.fromarray(im).resize((32, 32), Image.BILINEAR)) for im in images_raw])

    with open(DATA_DIR / "splits_10cv.json", "r", encoding="utf-8") as f:
        splits_info = json.load(f)

    splits = splits_info["splits_multiclass_stratified"]

    if single_fold:
        splits = splits[:1]
        print("Modo prueba rapida: Ejecutando solo Fold 0 para medir tiempo...")

    def create_baseline():
        clf = LeNet5BaselineClassifier(lr=0.01, epochs=25, batch_size=32, seed=42)
        clf.transform = None  # Las imagenes ya estan en 32x32, ToTensor directo
        return clf

    start_time = time.perf_counter()
    metrics_df, oof_df = evaluate_cv(
        model_name="baseline_lenet5",
        model_factory=create_baseline,
        images=images_32,
        labels=labels_bin,
        splits=splits,
        return_oof=True,
    )
    total_time = time.perf_counter() - start_time

    print(f"\nTiempo total de ejecucion: {total_time:.2f} s ({total_time / len(splits):.2f} s/pliegue)")

    if not single_fold:
        # Guardar resultados completos
        cv_path = RESULTS_DIR / "baseline_lenet5_cv_results.csv"
        oof_path = RESULTS_DIR / "baseline_lenet5_oof.csv"
        summary_path = RESULTS_DIR / "baseline_lenet5_summary.csv"

        metrics_df.to_csv(cv_path, index=False)
        oof_df.to_csv(oof_path, index=False)

        summary_df = summarize_results(metrics_df)
        summary_df.to_csv(summary_path)

        print("\n" + "=" * 60)
        print("RESUMEN ESTADISTICO 10-CV (Media ± Desv. Tipica):")
        print(summary_df.to_string())
        print("=" * 60)

        # Matriz de confusion global sobre OOF (las 5000 instancias)
        cm = metrics.confusion_matrix(oof_df["y_true"], oof_df["y_pred"])
        print("\nMatriz de confusion global (5.000 muestras OOF):")
        print("                     Pred: NO TUMOR (0)    Pred: TUMOR (1)")
        print(f"Real: NO TUMOR (0)          {cm[0, 0]:5d}               {cm[0, 1]:5d}")
        print(f"Real: TUMOR (1)             {cm[1, 0]:5d}               {cm[1, 1]:5d}")

        print("\nInforme de clasificacion global:")
        print(metrics.classification_report(oof_df["y_true"], oof_df["y_pred"], target_names=["NO TUMOR", "TUMOR"]))


if __name__ == "__main__":
    import sys
    only_one = "--benchmark" in sys.argv
    run_lenet5_baseline(single_fold=only_one)

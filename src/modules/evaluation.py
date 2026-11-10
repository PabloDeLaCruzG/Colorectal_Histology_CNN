"""Modulo de evaluacion y estadistica para el desafio de Histologia Colorrectal.

Replicado fielmente del estandar de nlp_challenge.py:
- compute_metrics: ROC-AUC continua, accuracy, F1 binario y macro.
- evaluate_cv: arnes 10-CV estratificado con fabrica de modelos limpios y predicciones OOF.
- summarize_results: media y desviacion tipica entre pliegues.
- wilcoxon_compare: test de rangos con signo pareado sobre los 10 pliegues.
- holm_bonferroni: correccion FWER step-down.
"""

from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import metrics


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
    """Calcula ROC-AUC con probabilidades continuas, accuracy y F1-score."""
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    # ROC-AUC continuo sobre la probabilidad de la clase positiva (1: TUMOR)
    # Si por alguna razon un pliegue tuviera una sola clase (imposible con 10-CV estratificada), control de excepcion
    try:
        auc = float(metrics.roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    acc = float(metrics.accuracy_score(y_true, y_pred))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    f1_macro = float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "f1": f1,
        "f1_macro": f1_macro,
    }


def evaluate_cv(
    model_name: str,
    model_factory: Callable,
    images: np.ndarray,
    labels: np.ndarray,
    splits: list[dict],
    return_oof: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena y evalua un modelo en cada pliegue 10-CV.

    Replicado del diseno de nlp_challenge.py:
    - model_factory(): genera una instancia limpia en cada pliegue (cero fuga entre pliegues).
    - OOF (Out of Fold): predicciones de test ensambladas para matriz de confusion global.
    """
    rows = []
    oof_rows = []

    print(f"\n{'=' * 60}")
    print(f"Evaluando modelo en 10-CV: {model_name}")
    print(f"{'=' * 60}")

    for fold_info in splits:
        k = fold_info["fold"]
        train_idx = np.array(fold_info["train_indices"])
        test_idx = np.array(fold_info["test_indices"])

        train_imgs, train_y = images[train_idx], labels[train_idx]
        test_imgs, test_y = images[test_idx], labels[test_idx]

        # Instanciar modelo limpio por pliegue
        model = model_factory()
        model.fit(train_imgs, train_y)

        # Predicciones continuas y discretas en test
        y_prob = model.predict_proba(test_imgs)
        y_pred = model.predict(test_imgs)

        fold_metrics = compute_metrics(test_y, y_prob, y_pred)
        row = {
            "model": model_name,
            "fold": k,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **fold_metrics,
        }
        rows.append(row)

        print(
            f"  Fold {k:2d} | ROC-AUC: {fold_metrics['roc_auc']:.4f} | "
            f"Accuracy: {fold_metrics['accuracy']:.4f} | F1: {fold_metrics['f1']:.4f}"
        )

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


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Media y desviacion tipica entre pliegues de ROC-AUC, Accuracy y F1 por modelo."""
    summary = results.groupby("model").agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )
    return summary.round(4)


def wilcoxon_compare(results: pd.DataFrame, model_a: str, model_b: str, metric: str = "roc_auc") -> dict:
    """Prueba de rangos con signo de Wilcoxon sobre las diferencias pareadas por pliegue.

    Replicado de nlp_challenge.py adaptado a ROC-AUC (donde mayor es mejor).
    """
    a = results[results["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = results[results["model"] == model_b].sort_values("fold")[metric].to_numpy()
    assert len(a) == len(b), "Los dos modelos deben tener los mismos pliegues"

    # En AUC y Accuracy, mayor es mejor (a > b)
    stat, p_value = wilcoxon(a, b)
    return {
        "metric": metric,
        "a": model_a,
        "b": model_b,
        "median_diff_a_minus_b": float(np.median(a - b)),
        "folds_a_better": int(np.sum(a > b)),
        "p_value": float(p_value),
    }


def holm_bonferroni(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Procedimiento step-down de Holm-Bonferroni para controlar el FWER.

    Replicado exactamente de nlp_challenge.py.
    """
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([], dtype=float)
    sort_idx = np.argsort(p_vals)
    p_sorted = p_vals[sort_idx]
    p_adj_sorted = np.zeros(m, dtype=float)
    cum_max = 0.0
    for i in range(m):
        rank = i + 1
        adj = (m - rank + 1) * p_sorted[i]
        cum_max = max(cum_max, adj)
        p_adj_sorted[i] = min(1.0, cum_max)
    p_adj = np.zeros(m, dtype=float)
    p_adj[sort_idx] = p_adj_sorted
    return p_adj

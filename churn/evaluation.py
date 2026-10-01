import matplotlib

matplotlib.use("Agg")  # no display needed, we only write files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def cross_validated_scores(pipeline, X, y, cv):
    """Manual CV loop so we get per-fold scores and the out-of-fold probabilities
    (used to pick the threshold) from a single pass over the folds."""
    oof = np.zeros(len(y))
    roc, pr = [], []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        roc.append(roc_auc_score(y.iloc[val_idx], proba))
        pr.append(average_precision_score(y.iloc[val_idx], proba))
    summary = {
        "cv_roc_auc_mean": float(np.mean(roc)),
        "cv_roc_auc_std": float(np.std(roc, ddof=1)),
        "cv_pr_auc_mean": float(np.mean(pr)),
        "cv_pr_auc_std": float(np.std(pr, ddof=1)),
    }
    return summary, oof


def best_f1_threshold(y_true, proba, window=5):
    """Threshold with the best F1 on a 0.01 grid.

    The F1 curve is fairly flat near its peak, so it is smoothed over a few
    neighbouring thresholds first; that avoids picking an isolated lucky spike.
    Because of the smoothing window the search effectively covers 0.07 - 0.93.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    if y_true.min() == y_true.max() or np.ptp(proba) == 0:
        raise ValueError("need both classes and non-constant scores to tune a threshold")
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1 = np.array([f1_score(y_true, proba >= t, zero_division=0) for t in grid])
    smooth = np.convolve(f1, np.ones(window) / window, mode="same")
    half = window // 2  # the moving average is padded at both ends, so skip those
    inner = slice(half, len(grid) - half)
    return float(grid[inner][np.argmax(smooth[inner])])


def classification_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_intervals(y_true, proba, threshold, n_boot=1000, seed=0):
    """95% percentile intervals for the headline test metrics. The test set is
    only ~1400 rows, so it is worth seeing how much of a difference is noise."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    draws = {"roc_auc": [], "pr_auc": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        yb, pb = y_true[idx], proba[idx]
        draws["roc_auc"].append(roc_auc_score(yb, pb))
        draws["pr_auc"].append(average_precision_score(yb, pb))
        draws["f1"].append(f1_score(yb, (pb >= threshold).astype(int), zero_division=0))
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k, v in draws.items()
    }


# ---------------------------------------------------------------- plots

def plot_roc_pr(curves, path):
    """curves: {model name: (y_true, proba)}, all on the same test set"""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    base_rate = float(np.mean(next(iter(curves.values()))[0]))
    for name, (y_true, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax_pr.plot(rec, prec, label=f"{name} ({average_precision_score(y_true, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax_pr.axhline(base_rate, color="k", ls="--", lw=0.8, label=f"no skill ({base_rate:.2f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test set)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_pr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(metrics, path):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["stays", "churns"],
           yticklabels=["stays", "churns"], xlabel="Predicted", ylabel="Actual",
           title=f"Test set, threshold {metrics['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_calibration(y_true, proba, path):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn rate",
           title="Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(names, means, stds, path, top=12):
    order = np.argsort(means)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order])
    ax.set(xlabel="Drop in PR-AUC when the column is shuffled",
           title="Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

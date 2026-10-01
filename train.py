"""Train, compare and export the churn model.

    python train.py            # full run (a couple of minutes)
    python train.py --quick    # smaller searches, writes to models_quick/ and reports_quick/

Everything the model learns, and every choice made (hyper-parameters, model,
threshold), comes from the training split. Test-set numbers are computed for
the report and never used to decide anything.
"""
import argparse
import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import StratifiedKFold

from churn import __version__
from churn.config import (
    CV_FOLDS,
    DATA_PATH,
    FEATURES,
    METADATA_FILENAME,
    MODEL_DIR,
    MODEL_FILENAME,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
)
from churn.data import load_raw, make_train_test_split, split_features_target
from churn.evaluation import (
    best_f1_threshold,
    bootstrap_intervals,
    classification_metrics,
    cross_validated_scores,
    plot_calibration,
    plot_confusion_matrix,
    plot_importance,
    plot_roc_pr,
)
from churn.modeling import fit_candidate, get_candidates, with_class_weights

log = logging.getLogger("train")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="path to the Telco CSV")
    parser.add_argument("--model-dir", type=Path, help="default: models/ (models_quick/ with --quick)")
    parser.add_argument("--report-dir", type=Path, help="default: reports/ (reports_quick/ with --quick)")
    parser.add_argument("--quick", action="store_true", help="small searches, for smoke tests")
    args = parser.parse_args(argv)

    # a smoke test must not overwrite the shipped model and reports
    suffix = "_quick" if args.quick else ""
    if args.model_dir is None:
        args.model_dir = MODEL_DIR.with_name(MODEL_DIR.name + suffix)
    if args.report_dir is None:
        args.report_dir = REPORT_DIR.with_name(REPORT_DIR.name + suffix)
    return args


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(rows, selected):
    header = ["Model", "CV ROC-AUC", "CV PR-AUC", "Test ROC-AUC", "Test PR-AUC",
              "Test F1", "Precision", "Recall", "Accuracy"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cv, t = r["cv"], r["test"]
        name = f"**{r['name']}**" if r["name"] == selected else r["name"]
        cells = [
            name,
            f"{cv['cv_roc_auc_mean']:.3f} +/- {cv['cv_roc_auc_std']:.3f}",
            f"{cv['cv_pr_auc_mean']:.3f} +/- {cv['cv_pr_auc_std']:.3f}",
            f"{t['roc_auc']:.3f}",
            f"{t['pr_auc']:.3f}",
            f"{t['f1']:.3f}",
            f"{t['precision']:.3f}",
            f"{t['recall']:.3f}",
            f"{t['accuracy']:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def imbalance_check(final_model, X_train, y_train, cv, unweighted_oof, unweighted_cv):
    """Same model, same hyper-parameters, same folds: with and without class
    weights. This is what backs the decision to keep the shipped model unweighted."""

    def summarise(summary, oof):
        thr = best_f1_threshold(y_train, oof)
        return {
            **summary,
            "brier": float(brier_score_loss(y_train, oof)),
            "best_threshold": thr,
            "f1_at_best_threshold": float(f1_score(y_train, oof >= thr)),
            "f1_at_0.5": float(f1_score(y_train, oof >= 0.5)),
        }

    weighted_cv, weighted_oof = cross_validated_scores(
        with_class_weights(final_model, y_train), X_train, y_train, cv
    )
    return {
        "unweighted": summarise(unweighted_cv, unweighted_oof),
        "class_weighted": summarise(weighted_cv, weighted_oof),
    }


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fig_dir = args.report_dir / "figures"
    for d in (args.model_dir, args.report_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.data)
    X, y = split_features_target(df)
    blank_total = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
    log.info("loaded %d customers, churn rate %.1f%%, %d blank TotalCharges values",
             len(df), 100 * y.mean(), blank_total)

    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted, oof_by_model, test_proba = [], {}, {}, {}
    for cand in get_candidates(quick=args.quick):
        start = time.perf_counter()
        model, best_params = fit_candidate(cand, X_train, y_train, cv)
        cv_summary, oof = cross_validated_scores(model, X_train, y_train, cv)

        # threshold comes from out-of-fold predictions on the training data, never the test set
        threshold = best_f1_threshold(y_train, oof) if cand.tune_threshold else 0.5
        proba = model.predict_proba(X_test)[:, 1]

        fitted[cand.name], oof_by_model[cand.name], test_proba[cand.name] = model, oof, proba
        rows.append({
            "name": cand.name,
            "best_params": to_jsonable(best_params),
            "cv": cv_summary,
            "threshold": threshold,
            "test": classification_metrics(y_test, proba, threshold),
        })
        log.info("%-22s CV PR-AUC %.3f  (%.0fs)", cand.name,
                 cv_summary["cv_pr_auc_mean"], time.perf_counter() - start)

    # Model selection looks at cross-validation on the training split only.
    contenders = [r for r in rows if r["name"] != "dummy_majority"]
    best = max(contenders, key=lambda r: r["cv"]["cv_pr_auc_mean"])
    name = best["name"]
    final_model = fitted[name]
    threshold = best["threshold"]
    log.info("selected %s (CV PR-AUC %.3f), decision threshold %.3f",
             name, best["cv"]["cv_pr_auc_mean"], threshold)

    table = markdown_table(rows, selected=name)
    log.info("\n%s", table)

    intervals = bootstrap_intervals(
        y_test.to_numpy(), test_proba[name], threshold,
        n_boot=200 if args.quick else 1000, seed=RANDOM_STATE,
    )
    imbalance = imbalance_check(
        final_model, X_train, y_train, cv, oof_by_model[name], best["cv"]
    )
    for label, r in imbalance.items():
        log.info("imbalance check %-15s PR-AUC %.3f  Brier %.3f  F1@0.5 %.3f  F1@best-thr %.3f",
                 label, r["cv_pr_auc_mean"], r["brier"], r["f1_at_0.5"], r["f1_at_best_threshold"])

    importance = permutation_importance(
        final_model, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )

    # ---- write everything out
    plot_roc_pr({r["name"]: (y_test, test_proba[r["name"]]) for r in contenders},
                fig_dir / "roc_pr_curves.png")
    plot_confusion_matrix(best["test"], fig_dir / "confusion_matrix.png")
    plot_calibration(y_test, test_proba[name], fig_dir / "calibration.png")
    plot_importance(FEATURES, importance.importances_mean, importance.importances_std,
                    fig_dir / "permutation_importance.png")

    (args.report_dir / "model_comparison.md").write_text(table + "\n", encoding="utf-8")
    metrics = {
        "selected_model": name,
        "selection_rule": "highest mean cross-validated PR-AUC on the training split",
        "models": rows,
        "test_intervals_95": intervals,
        "imbalance_check": imbalance,
        "permutation_importance": {
            f: {"mean": float(m), "std": float(s)}
            for f, m, s in zip(FEATURES, importance.importances_mean, importance.importances_std)
        },
    }
    (args.report_dir / "metrics.json").write_text(
        json.dumps(to_jsonable(metrics), indent=2), encoding="utf-8")

    metadata = {
        "model_name": name,
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": TARGET,
        "positive_label": "Yes",
        "features": FEATURES,
        "decision_threshold": threshold,
        "threshold_rule": "maximises F1 on out-of-fold predictions of the training split",
        "best_params": best["best_params"],
        "cv_metrics": best["cv"],
        "test_metrics": best["test"],
        "test_intervals_95": intervals,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_churn_rate": float(y_train.mean()),
        "random_state": RANDOM_STATE,
        "data_sha256": sha256_of(args.data),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(final_model, args.model_dir / MODEL_FILENAME, compress=3)
    (args.model_dir / METADATA_FILENAME).write_text(
        json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    log.info("saved model to %s", args.model_dir / MODEL_FILENAME)


if __name__ == "__main__":
    main()

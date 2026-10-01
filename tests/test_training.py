import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from churn.config import METADATA_FILENAME, MODEL_FILENAME, RANDOM_STATE
from churn.data import split_features_target
from churn.evaluation import cross_validated_scores
from churn.modeling import fit_candidate, get_candidates, with_class_weights
import train


def test_candidates_fit_and_score_on_a_small_sample(raw_df):
    X, y = split_features_target(raw_df.sample(800, random_state=0))
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    candidates = {c.name: c for c in get_candidates(quick=True)}

    for name in ("dummy_majority", "logreg_baseline", "xgboost_default"):
        model, params = fit_candidate(candidates[name], X, y, cv, n_jobs=1)
        assert params == {}
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),) and np.isfinite(proba).all()

    summary, oof = cross_validated_scores(model, X, y, cv)
    assert len(oof) == len(y)
    assert 0.5 < summary["cv_roc_auc_mean"] < 1.0


def test_quick_runs_do_not_default_to_the_shipped_folders():
    assert train.parse_args([]).model_dir.name == "models"
    assert train.parse_args(["--quick"]).model_dir.name == "models_quick"
    assert train.parse_args(["--quick"]).report_dir.name == "reports_quick"
    assert train.parse_args(["--quick", "--model-dir", "x"]).model_dir.name == "x"


def test_class_weights_only_touch_the_estimator(raw_df):
    X, y = split_features_target(raw_df)
    base = {c.name: c for c in get_candidates(quick=True)}
    lr = with_class_weights(base["logreg_baseline"].pipeline, y)
    xgb = with_class_weights(base["xgboost_default"].pipeline, y)
    assert lr.named_steps["model"].class_weight == "balanced"
    assert xgb.named_steps["model"].scale_pos_weight == pytest.approx((y == 0).sum() / (y == 1).sum())


@pytest.mark.slow
def test_quick_training_run_writes_all_artifacts(tmp_path):
    models, reports = tmp_path / "models", tmp_path / "reports"
    train.main(["--quick", "--model-dir", str(models), "--report-dir", str(reports)])

    assert (reports / "model_comparison.md").exists()
    assert (reports / "figures" / "roc_pr_curves.png").exists()
    metadata = json.loads((models / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert 0 < metadata["decision_threshold"] < 1
    assert metadata["test_metrics"]["roc_auc"] > 0.8
    pipeline = joblib.load(models / MODEL_FILENAME)
    assert hasattr(pipeline, "predict_proba")

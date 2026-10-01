from dataclasses import dataclass

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import RANDOM_STATE
from churn.preprocessing import TelcoCleaner, build_preprocessor

# Average precision is what the searches optimise: with ~27% positives it reacts
# much more to ranking churners near the top than ROC-AUC does.
SEARCH_SCORING = "average_precision"

XGB_FIXED = dict(
    random_state=RANDOM_STATE,
    n_jobs=1,  # parallelism happens across CV folds instead
    eval_metric="logloss",
    tree_method="hist",
)

XGB_SPACE = {
    "model__n_estimators": randint(150, 600),
    "model__learning_rate": loguniform(0.01, 0.15),
    "model__max_depth": randint(2, 6),
    "model__min_child_weight": randint(1, 12),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__reg_lambda": loguniform(0.5, 20),
}

RF_SPACE = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 8, 12, None],
    "model__min_samples_leaf": [1, 3, 5, 10],
    "model__max_features": ["sqrt", 0.3, 0.5],
}

LOGREG_SPACE = {"model__C": np.logspace(-2, 1, 7)}


@dataclass
class Candidate:
    name: str
    pipeline: Pipeline
    param_space: dict | None = None  # None means "fit as is"
    n_iter: int | None = None  # with a param_space: None = full grid, else random search
    tune_threshold: bool = True


def build_pipeline(estimator, scale_numeric):
    return Pipeline(
        [
            ("clean", TelcoCleaner()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def get_candidates(quick=False):
    rf_iter, xgb_iter = (6, 12) if quick else (15, 60)
    return [
        # floor: always predicts the majority class, so it shows what accuracy hides
        Candidate(
            "dummy_majority",
            build_pipeline(DummyClassifier(strategy="prior"), scale_numeric=False),
            tune_threshold=False,
        ),
        Candidate(
            "logreg_baseline",
            build_pipeline(LogisticRegression(max_iter=2000), scale_numeric=True),
        ),
        Candidate(
            "logreg_tuned",
            build_pipeline(LogisticRegression(max_iter=2000), scale_numeric=True),
            LOGREG_SPACE,
        ),
        Candidate(
            "random_forest_tuned",
            build_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                scale_numeric=False,
            ),
            RF_SPACE,
            rf_iter,
        ),
        Candidate(
            "xgboost_default",
            build_pipeline(XGBClassifier(**XGB_FIXED), scale_numeric=False),
        ),
        # the primary model
        Candidate(
            "xgboost_tuned",
            build_pipeline(XGBClassifier(**XGB_FIXED), scale_numeric=False),
            XGB_SPACE,
            xgb_iter,
        ),
    ]


def fit_candidate(candidate, X, y, cv, n_jobs=-1):
    """Fit on the training data only. Searches are cross-validated on the training
    folds and refit on all of the training data at the end."""
    if candidate.param_space is None:
        return clone(candidate.pipeline).fit(X, y), {}

    common = dict(scoring=SEARCH_SCORING, cv=cv, n_jobs=n_jobs, refit=True)
    if candidate.n_iter is None:
        search = GridSearchCV(candidate.pipeline, candidate.param_space, **common)
    else:
        search = RandomizedSearchCV(
            candidate.pipeline,
            candidate.param_space,
            n_iter=candidate.n_iter,
            random_state=RANDOM_STATE,
            **common,
        )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def with_class_weights(pipeline, y):
    """Copy of a pipeline with the estimator re-weighted towards the churn class.
    Only used for the imbalance check, not for the shipped model."""
    weighted = clone(pipeline)
    estimator = weighted.named_steps["model"]
    if isinstance(estimator, XGBClassifier):
        ratio = float((y == 0).sum() / (y == 1).sum())
        weighted.set_params(model__scale_pos_weight=ratio)
    elif isinstance(estimator, RandomForestClassifier):
        weighted.set_params(model__class_weight="balanced_subsample")
    else:
        weighted.set_params(model__class_weight="balanced")
    return weighted

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    BINARY_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NO_INTERNET_COLUMNS,
    NUMERIC_FEATURES,
)


def _clean_text(s):
    """Trim whitespace and turn blanks/None into NaN so the imputer can see them."""
    s = s.astype(object)
    s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    return s.mask(s.isna() | (s == ""), np.nan)


class TelcoCleaner(TransformerMixin, BaseEstimator):
    """Turns a raw customer frame (CSV row or JSON payload) into clean typed columns.

    Nothing here is learned from data, so it cannot leak anything between folds.
    It lives inside the Pipeline so that the saved model accepts raw input directly.

    - selects the 19 model features (customerID, Churn and any extras are dropped)
    - TotalCharges: " " / "29.85" / None -> float. A missing total is filled with
      tenure * MonthlyCharges; in the training data the only blanks are customers
      with tenure 0 (not billed yet), and that rule gives 0 for them.
    - "No internet service" / "No phone service" -> "No"
    """

    def fit(self, X, y=None):
        self._check_columns(X)
        return self

    def transform(self, X):
        self._check_columns(X)
        X = X[FEATURES].copy()

        for col in CATEGORICAL_FEATURES:
            X[col] = _clean_text(X[col])
        for col in NO_INTERNET_COLUMNS:
            X[col] = X[col].mask(X[col] == "No internet service", "No")
        X["MultipleLines"] = X["MultipleLines"].mask(
            X["MultipleLines"] == "No phone service", "No"
        )

        for col in NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES:
            X[col] = pd.to_numeric(_clean_text(X[col]), errors="coerce")

        X["TotalCharges"] = X["TotalCharges"].fillna(X["tenure"] * X["MonthlyCharges"])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)

    @staticmethod
    def _check_columns(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoCleaner expects a pandas DataFrame with the raw columns")
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing required columns: {missing}")


def build_preprocessor(scale_numeric=True):
    """Imputation + encoding. Everything in here is fitted, so it must only ever
    be fitted on training data; using it through a Pipeline takes care of that."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:  # trees don't care about scale, linear models do
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                BINARY_NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )

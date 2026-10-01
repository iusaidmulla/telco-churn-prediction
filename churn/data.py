import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import (
    DATA_PATH,
    FEATURES,
    ID_COL,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

log = logging.getLogger(__name__)


def load_raw(path=DATA_PATH):
    """Read the Kaggle CSV as-is. TotalCharges stays a string here on purpose;
    the cleaning step inside the pipeline is what converts it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and put the CSV at that location (or pass --data)."
        )
    df = pd.read_csv(path)

    missing = [c for c in [ID_COL, TARGET, *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    if df[ID_COL].duplicated().any():
        raise ValueError("customerID is supposed to be unique but has duplicates")
    return df


def split_features_target(df):
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError(f"{TARGET} has values other than Yes/No")
    # the ID is an identifier, not a signal, so it never reaches the model
    X = df[FEATURES].copy()
    return X, y.astype(int)


def make_train_test_split(X, y):
    """Stratified hold-out, done before anything is fitted. Everything learned later
    (imputers, encoders, hyper-parameters, threshold) uses only the training part."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(
        "split: %d train / %d test rows, churn rate %.3f / %.3f",
        len(X_train), len(X_test), y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test

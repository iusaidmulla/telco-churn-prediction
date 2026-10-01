import json
from pathlib import Path

import pandas as pd
import pytest
import sklearn

from churn.config import DATA_PATH, METADATA_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_payload():
    return json.loads((ROOT / "examples" / "sample_customer.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip("Telco CSV not available")
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def trained_model_dir():
    if not MODEL_PATH.exists():
        pytest.skip("no trained model yet, run `python train.py`")
    trained_with = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["versions"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        pytest.skip(f"models/ was trained with scikit-learn {trained_with}, found {sklearn.__version__}")
    return MODEL_PATH.parent

import json
import logging
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.exceptions import InconsistentVersionWarning

from churn.config import FEATURES, METADATA_FILENAME, MODEL_DIR, MODEL_FILENAME
from churn.schemas import CustomerProfile

log = logging.getLogger(__name__)


class ChurnPredictor:
    """Loads the saved pipeline once and scores single customer profiles.

    The saved pipeline outputs probabilities; the 0/1 decision uses the tuned
    threshold from model_metadata.json (calling .predict() on the bare pipeline
    would cut at 0.5 instead).
    """

    def __init__(self, pipeline, metadata):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata.get("decision_threshold", 0.5))

    @classmethod
    def load(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        model_path = model_dir / MODEL_FILENAME
        meta_path = model_dir / METADATA_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python train.py` first."
            )

        with warnings.catch_warnings():
            # the version check below gives a clearer message than sklearn's own warning
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            pipeline = joblib.load(model_path)

        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if not metadata:
            log.warning("no metadata next to the model, falling back to a 0.5 threshold")

        installed = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
        for package, current in installed.items():
            trained_with = metadata.get("versions", {}).get(package)
            if trained_with and trained_with != current:
                log.warning(
                    "model was trained with %s %s but %s is installed; install the "
                    "versions in requirements.txt or re-run `python train.py`",
                    package, trained_with, current,
                )

        predictor = cls(pipeline, metadata)
        predictor._smoke_test()
        return predictor

    def _smoke_test(self):
        # A pickle from another library version can load fine and still fail on the
        # first real row, so score one throwaway row now instead of on the first request.
        try:
            self.pipeline.predict_proba(pd.DataFrame(index=[0], columns=FEATURES))
        except Exception as exc:
            raise RuntimeError(
                f"the saved model cannot score with the installed libraries ({type(exc).__name__}: "
                f"{exc}). Install the versions in requirements.txt or re-run `python train.py`."
            ) from exc

    def predict(self, payload):
        """payload: a dict shaped like the raw customer JSON, or a CustomerProfile.
        Raises pydantic.ValidationError if the payload is not a valid profile."""
        profile = (
            payload
            if isinstance(payload, CustomerProfile)
            else CustomerProfile.model_validate(payload)
        )
        row = pd.DataFrame([profile.model_dump()])
        # decide on the same rounded number we return, so the output can never show
        # e.g. probability 0.34 next to a "No" when the threshold is 0.34
        probability = round(float(self.pipeline.predict_proba(row[FEATURES])[0, 1]), 4)
        churn = int(probability >= self.threshold)
        return {
            "customerID": profile.customerID,
            "churn_prediction": churn,
            "churn_label": "Yes" if churn else "No",
            "churn_probability": probability,
            "decision_threshold": round(self.threshold, 4),
        }

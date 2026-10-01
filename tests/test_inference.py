import json

import joblib
import pandas as pd
import pytest
from pydantic import ValidationError

import predict as predict_cli
from churn.config import MODEL_FILENAME
from churn.inference import ChurnPredictor


@pytest.fixture(scope="module")
def predictor(trained_model_dir):
    return ChurnPredictor.load(trained_model_dir)


def test_sample_payload_from_the_brief(predictor, sample_payload):
    out = predictor.predict(sample_payload)
    assert set(out) == {
        "customerID", "churn_prediction", "churn_label",
        "churn_probability", "decision_threshold",
    }
    assert out["customerID"] == "7590-VHVEG"
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["churn_prediction"] in (0, 1)
    assert out["churn_label"] == ("Yes" if out["churn_prediction"] else "No")
    assert out["churn_prediction"] == int(out["churn_probability"] >= out["decision_threshold"])


def test_prediction_is_deterministic(predictor, sample_payload):
    assert predictor.predict(sample_payload) == predictor.predict(sample_payload)


def test_total_charges_can_be_string_number_or_blank(predictor, sample_payload):
    base = predictor.predict(sample_payload)["churn_probability"]
    # tenure is 1, so a blank total is rebuilt as 1 * 29.85 and should score identically
    for total in (29.85, "29.85", " ", "", None):
        payload = {**sample_payload, "TotalCharges": total}
        assert predictor.predict(payload)["churn_probability"] == base


def test_customer_id_is_optional(predictor, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "customerID"}
    assert predictor.predict(payload)["customerID"] is None


def test_extra_keys_do_not_change_the_score(predictor, sample_payload):
    base = predictor.predict(sample_payload)["churn_probability"]
    noisy = {**sample_payload, "Churn": "Yes", "favourite_colour": "blue"}
    assert predictor.predict(noisy)["churn_probability"] == base


def test_high_risk_scores_above_low_risk(predictor, sample_payload):
    risky = {**sample_payload, "tenure": 2, "Contract": "Month-to-month",
             "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
             "MonthlyCharges": 95.0, "TotalCharges": "190"}
    safe = {**sample_payload, "tenure": 60, "Contract": "Two year",
            "PaymentMethod": "Bank transfer (automatic)", "TotalCharges": "1800"}
    assert predictor.predict(risky)["churn_probability"] > predictor.predict(safe)["churn_probability"]


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Weekly"},
        {"gender": "male"},
        {"tenure": -3},
        {"MonthlyCharges": "a lot"},
        {"SeniorCitizen": 2},
        {"TotalCharges": "n/a"},
        {"tenure": 10**12},  # absurd values are rejected, not scored
        {"tenure": 10**400},
        {"MonthlyCharges": 1e308},
        {"MonthlyCharges": float("inf")},
        {"TotalCharges": float("nan")},
        {"gender": 1},  # numbers are only coerced for the free-text customerID
    ],
)
def test_bad_values_are_rejected(predictor, sample_payload, change):
    with pytest.raises(ValidationError):
        predictor.predict({**sample_payload, **change})


def test_missing_field_is_rejected(predictor, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "Contract"}
    with pytest.raises(ValidationError, match="Contract"):
        predictor.predict(payload)


def test_saved_pipeline_takes_raw_dataframe_directly(trained_model_dir, raw_df):
    # what someone would do after a plain joblib.load(), string TotalCharges and all
    pipeline = joblib.load(trained_model_dir / MODEL_FILENAME)
    proba = pipeline.predict_proba(raw_df.head(50))[:, 1]
    assert proba.shape == (50,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_and_predictor_agree(predictor, sample_payload):
    direct = predictor.pipeline.predict_proba(pd.DataFrame([sample_payload]))[0, 1]
    assert predictor.predict(sample_payload)["churn_probability"] == round(float(direct), 4)


def test_cli_happy_path(trained_model_dir, tmp_path, sample_payload, capsys):
    path = tmp_path / "customer.json"
    path.write_text(json.dumps(sample_payload), encoding="utf-8")
    assert predict_cli.main([str(path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["customerID"] == "7590-VHVEG"


def test_saved_pipeline_predict_cuts_at_half_not_at_the_tuned_threshold(predictor, raw_df):
    # documents the behaviour described in the README: use ChurnPredictor / predict.py
    # for the 0/1 decision, the bare pipeline only gives probabilities (and .predict() at 0.5)
    X = raw_df.head(300)
    proba = predictor.pipeline.predict_proba(X)[:, 1]
    assert (predictor.pipeline.predict(X) == (proba >= 0.5)).all()


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_cli_reads_payload_files_written_by_windows_tools(
    trained_model_dir, tmp_path, sample_payload, capsys, encoding
):
    path = tmp_path / "customer.json"
    path.write_text(json.dumps(sample_payload), encoding=encoding)  # BOM / UTF-16 from Notepad, PowerShell
    assert predict_cli.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["customerID"] == "7590-VHVEG"


def test_cli_rejects_invalid_payload(trained_model_dir, capsys):
    assert predict_cli.main(["--json", json.dumps({"gender": "Other"})]) == 2
    assert "gender" in capsys.readouterr().err


def test_cli_handles_unreadable_payload(tmp_path, capsys):
    assert predict_cli.main([str(tmp_path / "nope.json")]) == 2
    assert "could not read" in capsys.readouterr().err


def test_cli_reports_corrupt_model_files(tmp_path, sample_payload, capsys):
    (tmp_path / MODEL_FILENAME).write_bytes(b"this is not a pickle")
    code = predict_cli.main(["--json", json.dumps(sample_payload), "--model-dir", str(tmp_path)])
    assert code == 1
    assert "could not load the model" in capsys.readouterr().err


def test_predictor_refuses_a_model_that_cannot_score(trained_model_dir, monkeypatch):
    # e.g. a pickle from another scikit-learn version that loads but fails on predict
    def broken(self, X):
        raise AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")

    monkeypatch.setattr("sklearn.pipeline.Pipeline.predict_proba", broken)
    with pytest.raises(RuntimeError, match="requirements.txt"):
        ChurnPredictor.load(trained_model_dir)


def test_cli_reports_missing_model(tmp_path, sample_payload, capsys):
    code = predict_cli.main(["--json", json.dumps(sample_payload), "--model-dir", str(tmp_path)])
    assert code == 1
    assert "train.py" in capsys.readouterr().err


def test_label_always_matches_the_returned_probability(predictor, raw_df):
    # the decision is made on the same rounded probability that is returned
    for row in raw_df.sample(300, random_state=0).to_dict(orient="records"):
        out = predictor.predict(row)
        assert out["churn_prediction"] == int(out["churn_probability"] >= out["decision_threshold"])


def test_numeric_customer_id_is_accepted_and_echoed(predictor, sample_payload):
    out = predictor.predict({**sample_payload, "customerID": 12345})
    assert out["customerID"] == "12345"

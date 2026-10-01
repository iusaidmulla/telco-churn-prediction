import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import DATA_PATH, FEATURES, NO_INTERNET_COLUMNS
from churn.data import load_raw, make_train_test_split, split_features_target
from churn.modeling import build_pipeline
from churn.preprocessing import TelcoCleaner, build_preprocessor


def one_row(sample_payload, **overrides):
    return pd.DataFrame([{**sample_payload, **overrides}])


def test_blank_total_charges_uses_tenure_times_monthly(sample_payload):
    rows = pd.concat(
        [
            one_row(sample_payload, tenure=0, TotalCharges=" "),  # how the CSV stores it
            one_row(sample_payload, tenure=10, MonthlyCharges=20.0, TotalCharges=""),
            one_row(sample_payload, tenure=5, MonthlyCharges=20.0, TotalCharges="123.4"),
            one_row(sample_payload, tenure=7, MonthlyCharges=10.0, TotalCharges=None),
        ],
        ignore_index=True,
    )
    total = TelcoCleaner().fit_transform(rows)["TotalCharges"]
    assert total.tolist() == [0.0, 200.0, 123.4, 70.0]
    assert total.dtype == float


def test_no_service_levels_are_folded_into_no(sample_payload):
    row = one_row(
        sample_payload,
        MultipleLines="No phone service",
        **{col: "No internet service" for col in NO_INTERNET_COLUMNS},
    )
    cleaned = TelcoCleaner().fit_transform(row)
    assert cleaned["MultipleLines"].iloc[0] == "No"
    assert (cleaned[NO_INTERNET_COLUMNS] == "No").all(axis=None)


def test_cleaner_keeps_only_model_features(sample_payload):
    row = one_row(sample_payload, Churn="Yes")
    cleaned = TelcoCleaner().fit_transform(row)
    assert list(cleaned.columns) == FEATURES


def test_cleaner_reports_missing_columns(sample_payload):
    row = one_row(sample_payload).drop(columns=["Contract", "tenure"])
    with pytest.raises(ValueError, match="Contract"):
        TelcoCleaner().fit_transform(row)


def test_whitespace_in_categories_is_trimmed(sample_payload):
    row = one_row(sample_payload, Contract=" Month-to-month ")
    assert TelcoCleaner().fit_transform(row)["Contract"].iloc[0] == "Month-to-month"


def test_real_data_is_clean_after_the_cleaner(raw_df):
    X, _ = split_features_target(raw_df)
    cleaned = TelcoCleaner().fit_transform(X)
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["TotalCharges"].dtype == float
    # the 11 blank strings in the CSV are all brand-new customers
    blank = X["TotalCharges"].astype(str).str.strip() == ""
    assert blank.sum() == 11
    assert (cleaned.loc[blank, "TotalCharges"] == 0).all()


def test_pipeline_survives_missing_and_unseen_values(raw_df):
    X, y = split_features_target(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale_numeric=True).fit(X, y)

    odd = X.head(3).copy()
    odd.loc[odd.index[0], "Contract"] = np.nan
    odd.loc[odd.index[1], "PaymentMethod"] = "Crypto"
    odd.loc[odd.index[2], "MonthlyCharges"] = np.nan
    with pytest.warns(UserWarning, match="unknown categories"):
        proba = pipe.predict_proba(odd)[:, 1]
    assert proba.shape == (3,)
    assert np.isfinite(proba).all()


def test_split_is_stratified_and_disjoint(raw_df):
    X, y = split_features_target(raw_df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_preprocessor_statistics_come_from_the_training_rows_only(raw_df):
    X, y = split_features_target(raw_df)
    X_train, X_test, y_train, _ = make_train_test_split(X, y)

    pipe = build_pipeline(LogisticRegression(max_iter=500), scale_numeric=True)
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    train_tenure_mean = X_train["tenure"].mean()
    assert scaler.mean_[0] == pytest.approx(train_tenure_mean)
    assert scaler.mean_[0] != pytest.approx(X["tenure"].mean())


def test_preprocessor_builds_without_scaling():
    assert "scale" not in build_preprocessor(scale_numeric=False).transformers[0][1].named_steps


def test_load_raw_accepts_a_plain_string_path(raw_df):
    assert len(load_raw(str(DATA_PATH))) == len(raw_df)

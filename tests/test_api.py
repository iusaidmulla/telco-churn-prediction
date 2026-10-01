import json

import pytest
from fastapi.testclient import TestClient

from churn.api import app


@pytest.fixture(scope="module")
def client(trained_model_dir):
    with TestClient(app) as c:  # the context manager runs the lifespan (model load)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert 0 < body["decision_threshold"] < 1


def test_predict_returns_label_and_probability(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["customerID"] == "7590-VHVEG"
    assert body["churn_label"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_invalid_category_gets_422_with_the_field_name(client, sample_payload):
    response = client.post("/predict", json={**sample_payload, "Contract": "Weekly"})
    assert response.status_code == 422
    assert "Contract" in str(response.json()["detail"])


def test_missing_field_gets_422(client, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_body_gets_422(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_non_finite_json_tokens_get_422_not_500(client, sample_payload, token):
    # Python's json parser accepts these, and FastAPI's default 422 body then can't be serialised
    body = json.dumps({**sample_payload, "tenure": "__x__"}).replace('"__x__"', token)
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "tenure" in str(response.json()["detail"])


def test_absurd_values_get_422(client, sample_payload):
    payload = {**sample_payload, "tenure": 72, "MonthlyCharges": 1e308, "TotalCharges": None}
    assert client.post("/predict", json=payload).status_code == 422

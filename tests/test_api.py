import pytest
from fastapi.testclient import TestClient
from fraud_detection.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan: model + store load
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_hard_cap_blocks_before_model(client):
    """Over-cap txn must BLOCK on the rule and never reach the model (prob == -1.0)."""
    r = client.post(
        "/score",
        json={
            "transaction_type": "TRANSFER",
            "amount": 99_000_000,
            "name_orig": "C1",
            "name_dest": "C2",
            "step": 2,
        },
    )
    body = r.json()
    assert body["decision"] == "BLOCK"
    assert body["rule_triggered"] == "hard_amount_cap"
    assert body["fraud_probability"] == -1.0  # sentinel: model not called


def test_normal_payment_scored(client):
    r = client.post(
        "/score",
        json={
            "transaction_type": "PAYMENT",
            "amount": 100,
            "name_orig": "C1",
            "name_dest": "M2",
            "step": 10,
        },
    )
    body = r.json()
    assert body["decision"] in {"ALLOW", "STEP_UP_AUTH", "HOLD", "BLOCK"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["latency_ms"] >= 0


def test_invalid_input_rejected(client):
    """Negative amount violates the schema -> 422."""
    r = client.post(
        "/score",
        json={
            "transaction_type": "TRANSFER",
            "amount": -5,
            "name_orig": "C1",
            "name_dest": "C2",
            "step": 1,
        },
    )
    assert r.status_code == 422

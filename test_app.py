
from app import app


def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_prediction():
    client = app.test_client()
    response = client.get("/predict?hours=5")
    assert response.status_code == 200
    assert "predicted_marks" in response.json


def test_invalid_hours():
    client = app.test_client()
    response = client.get("/predict?hours=30")
    assert response.status_code == 400

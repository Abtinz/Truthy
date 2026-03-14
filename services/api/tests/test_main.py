from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Truthy API",
        "status": "running",
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_review_submission_endpoint() -> None:
    response = client.post("/review")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Review endpoint connected",
    }


def test_review_lookup_endpoint() -> None:
    response = client.get("/review/review-123")

    assert response.status_code == 200
    assert response.json() == {
        "review_id": "review-123",
        "status": "pending",
    }


def test_policy_refresh_endpoint() -> None:
    response = client.post("/policy/refresh")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Policy refresh triggered",
    }

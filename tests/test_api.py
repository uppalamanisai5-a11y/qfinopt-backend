import pytest
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Q-FinOpt API"

def test_platform_recommendation():
    response = client.get("/api/platform/recommendation?investment=5000")
    assert response.status_code == 200
    data = response.json()
    assert data["top_platform"] == "Groww"
    assert len(data["platforms"]) >= 3

    response2 = client.get("/api/platform/recommendation?investment=500000")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["top_platform"] == "Zerodha Coin"

def test_nav_search():
    response = client.get("/api/nav/search?q=SBI")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_funds_in_amfi" in data

"""
Shared pytest fixtures for API tests.
Uses FastAPI TestClient (no live server required).
"""
import pytest
from starlette.testclient import TestClient

from apps.api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

"""Live integration tests against deployed Memwright Lambda.

Skip without MEMWRIGHT_LAMBDA_URL env var.
Usage: MEMWRIGHT_LAMBDA_URL=https://xxx.execute-api.us-east-1.amazonaws.com pytest tests/test_lambda_live.py -v
"""

import os
import uuid

import pytest
import requests

BASE_URL = os.environ.get("MEMWRIGHT_LAMBDA_URL", "")

pytestmark = pytest.mark.skipif(not BASE_URL, reason="MEMWRIGHT_LAMBDA_URL not set")


def url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


class TestHealth:
    def test_health_ok(self):
        r = requests.get(url("/health"), timeout=30)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True

    def test_stats(self):
        r = requests.get(url("/stats"), timeout=15)
        assert r.status_code == 200
        assert r.json()["ok"] is True


class TestCRUD:
    def test_add_and_recall(self):
        tag = f"test-{uuid.uuid4().hex[:8]}"
        # Add
        r = requests.post(url("/add"), json={
            "content": f"Lambda test memory {tag}",
            "tags": [tag],
            "category": "test",
        }, timeout=15)
        assert r.status_code == 200
        data = r.json()["data"]
        memory_id = data["id"]

        # Recall
        r = requests.post(url("/recall"), json={
            "query": f"Lambda test {tag}",
        }, timeout=15)
        assert r.status_code == 200
        results = r.json()["data"]
        found_ids = [res["id"] for res in results]
        assert memory_id in found_ids

    def test_get_by_id(self):
        # Add
        r = requests.post(url("/add"), json={
            "content": "get-by-id test",
            "tags": ["gettest"],
        }, timeout=15)
        mid = r.json()["data"]["id"]

        # Get
        r = requests.get(url(f"/memory/{mid}"), timeout=15)
        assert r.status_code == 200
        assert r.json()["data"]["content"] == "get-by-id test"

    def test_search(self):
        r = requests.post(url("/search"), json={
            "category": "test",
            "limit": 5,
        }, timeout=15)
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_forget(self):
        r = requests.post(url("/add"), json={
            "content": "to be forgotten",
            "tags": ["forget"],
        }, timeout=15)
        mid = r.json()["data"]["id"]

        r = requests.post(url("/forget"), json={"memory_id": mid}, timeout=15)
        assert r.status_code == 200
        assert r.json()["data"]["forgotten"] is True


class TestErrors:
    def test_add_missing_content(self):
        r = requests.post(url("/add"), json={}, timeout=15)
        assert r.status_code == 400
        assert r.json()["ok"] is False

    def test_recall_missing_query(self):
        r = requests.post(url("/recall"), json={}, timeout=15)
        assert r.status_code == 400

    def test_get_nonexistent(self):
        r = requests.get(url("/memory/nonexistent999"), timeout=15)
        assert r.status_code == 404

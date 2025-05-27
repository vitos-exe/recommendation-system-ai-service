from dataclasses import asdict, astuple

from flask import Flask
from flask.testing import FlaskClient
import pytest
from qdrant_client import QdrantClient

from app.db import COLLECTION_NAME
from app.model import Prediction, Track
from tests.db_base import TestDBBase


class TestEndpoint(TestDBBase):
    @pytest.fixture
    def api_client(self, app: Flask):
        return app.test_client()

    def test_populate_db(self, api_client: FlaskClient, db_client: QdrantClient):
        response = api_client.get("/populate_db")
        assert response.status_code == 204
        count = db_client.count(COLLECTION_NAME).count
        assert count == 100
        records = db_client.scroll(COLLECTION_NAME, limit=100, with_vectors=True)[0]
        for record in records:
            assert abs(1 - sum(record.vector)) < 1e-5
            assert len(record.payload) > 0

    def test_get_prediction(self, api_client: FlaskClient, db_client: QdrantClient):
        raw_lyrics = Track("artist", "title", "This is so sad")
        response = api_client.post("/predict?save=true", json=[asdict(raw_lyrics)])
        assert response.status_code == 200
        json = response.get_json()
        pred = Prediction(**json[0])
        assert max(astuple(pred)) == pred.sad
        records = db_client.scroll(COLLECTION_NAME)[0]
        assert len(records) == 1
        assert records[0].payload["artist"] == "artist"


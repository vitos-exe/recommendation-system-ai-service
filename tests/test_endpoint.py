from dataclasses import asdict, astuple

import pytest

from app.db import COLLECTION_NAME
from app.model import Prediction, Track
from tests.db_base import TestDBBase


class TestEndpoint(TestDBBase):
    @pytest.fixture
    def api_client(self, app):
        return app.test_client()

    def test_get_prediction(self, api_client, db_client):
        raw_lyrics = Track("artist", "title", "This is so sad")
        response = api_client.post("/predict?save=true", json=[asdict(raw_lyrics)])
        assert response.status_code == 200
        json = response.get_json()
        pred = Prediction(**json[0])
        assert max(astuple(pred)) == pred.sad
        records = db_client.scroll(COLLECTION_NAME)[0]
        assert len(records) == 1
        assert records[0].payload["artist"] == "artist"

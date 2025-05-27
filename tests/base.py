from random import random

import pytest

from ai_service import create_app
from ai_service.config import TestConfig
from ai_service.lyrics_reader import read_lyrics
from ai_service.model import PredictionTrack, Prediction, Track


class TestBase:
    TEST_PREDICTION = Prediction(random(), random(), random(), random())
    TEST_LYRICS = PredictionTrack("artist", "title", TEST_PREDICTION)
    TEST_RAW_LYRICS = Track("artist", "title", "lyrics")

    @pytest.fixture(scope="session")
    def app(self):
        app = create_app(TestConfig)
        return app

    @pytest.fixture
    def raw_lyrics(self, app):
        with app.app_context():
            raw_lyrics = read_lyrics()
            return raw_lyrics

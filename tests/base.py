from random import random

import pytest

from app import create_app
from app.config import TestConfig
from app.lyrics_reader import read_lyrics
from app.model import Prediction, PredictionTrack, Track


class TestBase:
    TEST_PREDICTION = Prediction(random(), random(), random(), random())
    TEST_LYRICS = PredictionTrack("artist", "title", TEST_PREDICTION)
    TEST_RAW_LYRICS = Track("artist", "title", "lyrics")

    @pytest.fixture(scope="function")
    def app(self):
        app = create_app(TestConfig())
        return app

    @pytest.fixture
    def raw_lyrics(self, app):
        with app.app_context():
            raw_lyrics = read_lyrics()
            return raw_lyrics

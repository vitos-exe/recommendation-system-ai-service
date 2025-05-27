from dataclasses import astuple

from testfixtures import compare

from app.ml import get_sentiment_model
from tests.base import TestBase


class TestML(TestBase):
    def test_predict_lyrics(self, app, raw_lyrics):
        with app.app_context():
            model = get_sentiment_model()
            predictions = model.predict_lyrics(raw_lyrics)
            prediction_sums = [round(sum(astuple(p)), 5) for p in predictions]
            compare([1] * len(predictions), prediction_sums)

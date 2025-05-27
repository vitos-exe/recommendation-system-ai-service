from dataclasses import astuple

from testfixtures import compare

from ai_service.db import COLLECTION_NAME, add_lyrics, search_n_closest
from ai_service.model import PredictionTrack, Prediction
from tests.base import TestBase
from tests.db_base import TestDBBase


class TestDB(TestDBBase):
    @staticmethod
    def round_to_five(f):
        return round(f, 5)

    def test_add_lyrics(self, app, db_client):  # Add app fixture
        with app.app_context():  # Add app context
            add_lyrics([TestBase.TEST_LYRICS])
            records = db_client.scroll(
                collection_name=COLLECTION_NAME, with_vectors=True
            )[0]
        assert len(records) == 1
        assert records[0].payload["artist"] == TestBase.TEST_LYRICS.artist
        vector = records[0].vector
        test_lyrics_vector = astuple(TestBase.TEST_LYRICS.prediction)
        compare(
            map(TestDB.round_to_five, vector),
            map(TestDB.round_to_five, test_lyrics_vector),
        )

    def test_n_closest(self, app, db_client):
        with app.app_context():
            preds = [
                [0.7, 0.1, 0.15, 0.05],
                [0.1, 0.6, 0.1, 0.2],
                [0.3, 0.1, 0.5, 0.1],
                [0.2, 0.2, 0.1, 0.5],
                [0.2, 0.5, 0.2, 0.1],
            ]
            titles = [
                "Happy Dance",
                "Sad Ballad",
                "Energetic Rock",
                "Relaxing Jazz",
                "Melancholy Pop",
            ]
            lyrics = [
                PredictionTrack("artist", t, Prediction(*p)) for t, p in zip(titles, preds)
            ]
            add_lyrics(lyrics)
            closest = search_n_closest(Prediction(0.6, 0.15, 0.2, 0.05), n=3, initial_n_candidates=3)
        compare([titles[0], titles[2], titles[4]], [c.title for c in closest])

from dataclasses import asdict
from json import dumps

from app.model import PredictionTrack
from tests.base import TestBase


class TestModel(TestBase):
    def test_dict_without_prediction(self):
        lyrics_dict = TestBase.TEST_LYRICS.dict_without_prediction
        assert (
            "artist" in lyrics_dict
            and "title" in lyrics_dict
            and "prediction" not in lyrics_dict
        )

    def test_combine_raw_lyrics_and_prediction(self):
        lyrics = PredictionTrack.get_from_track_and_prediction(
            TestBase.TEST_RAW_LYRICS, TestBase.TEST_PREDICTION
        )
        assert dumps(asdict(lyrics)) == dumps(asdict(TestBase.TEST_LYRICS))

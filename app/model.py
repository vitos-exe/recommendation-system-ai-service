from dataclasses import asdict, dataclass


@dataclass
class Prediction:
    angry: float
    happy: float
    relaxed: float
    sad: float


@dataclass
class TrackBase:
    artist: str
    title: str


@dataclass
class PredictionTrack(TrackBase):
    prediction: Prediction

    @property
    def dict_without_prediction(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "prediction"}

    @staticmethod
    def from_track_and_prediction(
        track: TrackBase, prediction: Prediction
    ) -> "PredictionTrack":
        return PredictionTrack(
            artist=track.artist,
            title=track.title,
            prediction=prediction,
        )


@dataclass
class Track(TrackBase):
    lyrics: str

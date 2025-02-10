from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Prediction:
    angry: float
    happy: float
    relaxed: float
    sad: float


@dataclass(frozen=True)
class Lyrics:
    artist: str
    title: str
    prediction: Prediction

    @property
    def dict_without_prediction(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "prediction"}


@dataclass(frozen=True)
class RawLyrics:
    artist: str
    title: str
    lyrics: str


def combine_raw_lyrics_and_prediction(
    raw_lyrics: RawLyrics, prediction: Prediction
) -> Lyrics:
    return Lyrics(
        artist=raw_lyrics.artist, title=raw_lyrics.title, prediction=prediction
    )

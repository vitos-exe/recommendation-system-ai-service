import uuid
from dataclasses import astuple

from flask import current_app
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from app.model import Prediction, PredictionTrack

COLLECTION_NAME: str = "lyrics"


class QdrantClientNotInitializedError(Exception):
    """Custom exception for when QdrantClient is accessed before initialization."""

    pass


def lyrics_to_point_struct(lyrics: PredictionTrack) -> PointStruct:
    return PointStruct(
        id=str(uuid.uuid4()),
        vector=astuple(lyrics.prediction),
        payload=lyrics.dict_without_prediction,
    )


def score_point_to_lyrics(point: ScoredPoint) -> PredictionTrack:
    pred = Prediction(*point.vector)
    return PredictionTrack(prediction=pred, **point.payload)


def create_qdrant_client(config) -> QdrantClient:
    location = config.get("QDRANT_URL")
    client = QdrantClient(location)
    setup_qdrant(client)
    return client


def setup_qdrant(client: QdrantClient) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4, distance=Distance.MANHATTAN),
        )


def get_qdrant_client() -> QdrantClient:
    client = current_app.extensions.get("qdrant_client")
    if client is None:
        raise QdrantClientNotInitializedError(
            "QdrantClient was accessed but it is None or not initialized in app.extensions"
        )
    return client


def add_lyrics(lyrics: list[PredictionTrack]) -> None:
    client = get_qdrant_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[lyrics_to_point_struct(lyr) for lyr in lyrics],
    )


def search_closest(pred: Prediction, n: int = 10) -> list[PredictionTrack]:
    client = get_qdrant_client()
    points = client.query_points(
        COLLECTION_NAME, list(astuple(pred)), limit=n, with_vectors=True
    ).points
    return [score_point_to_lyrics(p) for p in points]

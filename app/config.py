import os
from typing import Optional


class Config:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    LYRICS_FOLDER_STRUCTURE_PATH: str = os.getenv(
        "LYRICS_FOLDER_STRUCTURE_PATH", "lyrics.zip"
    )
    TESTING: bool = os.getenv("TESTING", "False").lower() == "false"
    MODELS_DIR: str = os.getenv("MODELS_DIR", "models")
    WORD2VEC_NAME: str = os.getenv("WORD2VEC_NAME", "word2vec-google-news-300")
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "TFIDF_SVM")


class DevConfig(Config):
    pass


class TestConfig(Config):
    QDRANT_URL = ":memory:"
    LYRICS_FOLDER_STRUCTURE_PATH = "tests/data/lyrics_test.zip"
    TESTING = True
    MODELS_DIR: str = "tests/models"

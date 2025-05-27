from abc import ABC, abstractmethod

from flask import current_app

from app.model import Prediction, Track


class SentimentModel(ABC):
    @abstractmethod
    def predict_lyrics(self, lyrics: list[Track]) -> list[Prediction]:
        pass


def create_model(app_config) -> SentimentModel:
    model_type = app_config.get("MODEL_TYPE", "LSTM")
    if model_type == "LSTM":
        from app.ml.lstm import SentimentLSTM

        return SentimentLSTM(app_config)
    elif model_type == "TFIDF_SVM":
        from app.ml.svm import SentimentTfIdfSVM

        return SentimentTfIdfSVM(app_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_sentiment_model() -> SentimentModel:
    model = current_app.extensions.get("sentiment_model")
    if model is None:
        raise Exception(
            "SentimentModel was accessed but it is None or not initialized in app.extensions"
        )
    return model

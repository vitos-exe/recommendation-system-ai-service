from abc import ABC, abstractmethod
from functools import cached_property

import joblib
import numpy as np
from flask import current_app

from ai_service.model import Prediction, Track
from ai_service.preprocessing_utils import preprocess


class SentimentModel(ABC):
    @abstractmethod
    def predict_lyrics(self, lyrics: list[Track]) -> list[Prediction]:
        pass


def create_model(app_config) -> SentimentModel:
    model_type = app_config.get("MODEL_TYPE", "LSTM")
    if model_type == "LSTM":
        from ai_service.nn_definition import SentimentLSTMCore
        return SentimentLSTM(app_config, SentimentLSTMCore)
    elif model_type == "TFIDF_SVM":
        return SentimentTfidfSvm(app_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class SentimentLSTM(SentimentModel):
    MODEL_FILE_NAME = "lstm_sentiment_model.pt"

    def __init__(self, app_config, SentimentLSTMCore_cls):
        import torch
        self.model = SentimentLSTMCore_cls(hidden_dim=128, n_layers=2)
        models_dir = app_config.get("MODELS_DIR", "models")
        model_path = f"{models_dir}/{self.MODEL_FILE_NAME}"
        self.model.load_state_dict(
            torch.load(
                model_path,
                weights_only=True,
            )
        )
        self._app_config = app_config
        if not app_config.get("TESTING", False):
            self.__dict__["word2vec"] = self.word2vec

    @cached_property
    def word2vec(self):
        word2vec_name = self._app_config.get(
            "WORD2VEC_NAME", "word2vec-google-news-300"
        )
        from gensim.downloader import load
        return load(word2vec_name)

    def get_sentence_embedding(
        self, sentence: str, vector_size: int = 300, max_len: int = 64
    ) -> np.ndarray:
        tokens = preprocess(sentence)
        vectors = [self.word2vec[word] for word in tokens if word in self.word2vec]
        if len(vectors) > max_len:
            vectors = vectors[:max_len]
        elif len(vectors) < max_len:
            vectors.extend([np.zeros(vector_size)] * (max_len - len(vectors)))

        if not vectors:
            return np.zeros((max_len, vector_size))
        return np.array(vectors)

    def predict_lyrics(self, lyrics: list[Track]) -> list[Prediction]:
        import torch
        embeddings = np.array([self.get_sentence_embedding(lr.lyrics) for lr in lyrics])
        X = torch.from_numpy(embeddings).float()
        preds = self.model(X).detach().numpy()
        return [Prediction(*[p.item() for p in pred]) for pred in preds]


class SentimentTfidfSvm(SentimentModel):
    TFIDF_VECTORIZER_FILE_NAME = "tfidf_vectorizer.pkl"
    SVM_CLASSIFIER_FILE_NAME = "svc_model.pkl"

    def __init__(self, app_config):
        models_dir = app_config.get("MODELS_DIR", "models")
        tfidf_path = f"{models_dir}/{self.TFIDF_VECTORIZER_FILE_NAME}"
        svm_path = f"{models_dir}/{self.SVM_CLASSIFIER_FILE_NAME}"

        with open(tfidf_path, "rb") as f:
            self.tfidf_vectorizer = joblib.load(f)
        with open(svm_path, "rb") as f:
            self.svm_classifier = joblib.load(f)

    def predict_lyrics(self, lyrics: list[Track]) -> list[Prediction]:
        lyrics_text = [" ".join(preprocess(lr.lyrics)) for lr in lyrics]
        lyrics_tfidf = self.tfidf_vectorizer.transform(lyrics_text).toarray()

        try:
            preds_proba = self.svm_classifier.predict_proba(lyrics_tfidf)
        except AttributeError:
            raw_preds = self.svm_classifier.predict(lyrics_tfidf)
            num_classes = 4
            if hasattr(self.svm_classifier, "classes_"):
                num_classes = len(self.svm_classifier.classes_)

            preds_proba = np.zeros((len(raw_preds), num_classes))
            for i, label in enumerate(raw_preds):
                label_idx = int(label)
                if 0 <= label_idx < num_classes:
                    preds_proba[i, label_idx] = 1.0
                else:
                    current_app.logger.warning(
                        f"Unexpected label {label} from SVM classifier."
                    )

        return [
            Prediction(angry=p[0], happy=p[1], relaxed=p[2], sad=p[3])
            for p in preds_proba
        ]


def get_sentiment_model() -> SentimentModel:
    model = current_app.extensions.get("sentiment_model")
    if model is None:
        raise Exception(
            "SentimentModel was accessed but it is None or not initialized in app.extensions"
        )
    return model

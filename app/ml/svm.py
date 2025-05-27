import joblib
import numpy as np
from flask import current_app

from app.ml import SentimentModel
from app.ml.preprocessing import preprocess
from app.model import Prediction, Track


class SentimentTfIdfSVM(SentimentModel):
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

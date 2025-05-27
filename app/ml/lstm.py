from functools import cached_property

import numpy as np
import torch
from torch import nn

from app.ml import SentimentModel
from app.ml.preprocessing import preprocess
from app.model import Prediction, Track


class SentimentLSTMCore(nn.Module):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 256,
        output_dim: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(SentimentLSTMCore, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return nn.softmax(logits, dim=1)


class SentimentLSTM(SentimentModel):
    MODEL_FILE_NAME = "lstm_sentiment_model.pt"

    def __init__(self, app_config):
        self.model = SentimentLSTMCore(hidden_dim=128, n_layers=2)
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
        embeddings = np.array([self.get_sentence_embedding(lr.lyrics) for lr in lyrics])
        X = torch.from_numpy(embeddings).float()
        preds = self.model(X).detach().numpy()
        return [Prediction(*[p.item() for p in pred]) for pred in preds]

import secrets
from dataclasses import asdict

from flask import Flask, request, Response

from app import db, lyrics_reader, ml, model, reranking
from app.config import Config, DevConfig

def create_app(app_config: Config=DevConfig()):
    app = Flask(__name__)
    app.secret_key = secrets.token_hex()
    app.config.from_object(app_config)
    app.config.from_prefixed_env()

    with app.app_context():
        if not hasattr(app, "extensions"):
            app.extensions = {}
        app.extensions["qdrant_client"] = db.create_qdrant_client(app.config)
        app.extensions["sentiment_model"] = ml.create_model(app.config)


    @app.get("/health")
    def health_check():
        return {"status": "healthy"}, 200

    @app.get("/populate_db")
    def populate_db():
        sentiment_model = ml.get_sentiment_model()
        raw_lyrics = lyrics_reader.read_lyrics()
        predictions = sentiment_model.predict_lyrics(raw_lyrics)
        lyrics = [
            model.PredictionTrack.get_from_track_and_prediction(rl, p)
            for rl, p in zip(raw_lyrics, predictions)
        ]
        db.add_lyrics(lyrics)
        return Response(status=204)

    @app.post("/prediction")
    def get_predictions():
        sentiment_model = ml.get_sentiment_model()
        save = request.args.get("save")
        lyrics = [model.Track(**obj) for obj in request.get_json()]
        preds = sentiment_model.predict_lyrics(lyrics)
        if save:
            tracks = [
                model.PredictionTrack.get_from_track_and_prediction(lyr, pred)
                for lyr, pred in zip(lyrics, preds)
            ]
            db.add_lyrics(tracks)
        return [asdict(p) for p in preds], 200

    @app.post("/closest")
    def get_closest():
        pred = model.Prediction(**request.get_json())
        n = request.args.get("n", default=1, type=int)
        initial_candidates = db.search_closest(pred, n * 5)
        reranked_candidates = reranking.rerank_lyrics(initial_candidates, pred)
        final_results = reranked_candidates[:n]
        return [asdict(t) for t in final_results], 200

    return app

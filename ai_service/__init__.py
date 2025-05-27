import secrets
from dataclasses import asdict

from flask import Flask, jsonify, request

from ai_service import db, lyrics_reader, ml, model, reranking


def create_app(app_config="ai_service.config.DevConfig"):
    app = Flask(__name__)
    app.secret_key = secrets.token_hex()
    app.config.from_object(app_config)
    app.config.from_prefixed_env()

    # Initialize extensions within the app context
    with app.app_context():
        if not hasattr(app, "extensions"):
            app.extensions = {}
        app.extensions["qdrant_client"] = db.create_qdrant_client(app.config)
        app.extensions["sentiment_model"] = ml.create_model(app.config)

    @app.cli.command("populate_db")
    def populate_db():
        sentiment_model = ml.get_sentiment_model()
        raw_lyrics = lyrics_reader.read_lyrics()
        predictions = sentiment_model.predict_lyrics(raw_lyrics)
        lyrics = [
            model.PredictionTrack.get_from_track_and_prediction(rl, p)
            for rl, p in zip(raw_lyrics, predictions)
        ]
        db.add_lyrics(lyrics)

    @app.get("/health")
    def health_check():
        return jsonify(status="healthy"), 200

    @app.post("/")
    def get_prediction():
        sentiment_model = ml.get_sentiment_model()
        save = request.args.get("save")
        raw_lyrics = model.Track(**request.get_json())
        prediction = sentiment_model.predict_lyrics([raw_lyrics])[0]
        if save:
            db.add_lyrics(
                [model.PredictionTrack.get_from_track_and_prediction(raw_lyrics, prediction)]
            )
        return asdict(prediction), 200

    @app.post("/batch")
    def get_predictions():
        sentiment_model = ml.get_sentiment_model()
        save = request.args.get("save")
        lyrics = [model.Track(**obj) for obj in request.get_json()]
        preds = sentiment_model.predict_lyrics(lyrics)
        if save:
            db.add_lyrics(
                [
                    model.PredictionTrack.get_from_track_and_prediction(lyr, pred)
                    for lyr, pred in zip(lyrics, preds)
                ]
            )
        return [asdict(p) for p in preds], 200

    @app.post("/closest")
    def get_closest():
        pred = model.Prediction(**request.get_json())
        # Define how many initial candidates to fetch and how many final results to return
        num_final_results = request.args.get("n", default=1, type=int)
        num_initial_candidates = request.args.get("initial_n", default=10, type=int)

        if num_final_results > num_initial_candidates:
            # Ensure we fetch enough candidates for the final selection
            num_initial_candidates = num_final_results * 2 

        initial_candidates = db.search_n_closest(pred, n=num_initial_candidates)
        
        # Apply reranking
        reranked_candidates = reranking.rerank_lyrics(initial_candidates, pred)

        # Select the top N results after reranking
        final_results = reranked_candidates[:num_final_results]
        
        return jsonify(final_results)

    return app

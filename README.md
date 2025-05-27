# Recommendation System AI Service

A music recommendation system that analyzes song lyrics to determine emotional sentiment and provides music recommendations based on emotional similarity.

## Overview

This service uses machine learning models to analyze the sentiment of song lyrics and categorize them into four emotional states:
- Angry
- Happy
- Relaxed
- Sad

Based on this sentiment analysis, the system can recommend similar songs that match a specified emotional profile. The service uses vector similarity search to efficiently find songs with similar emotional patterns.

## Architecture

The service consists of several key components:

1. **Flask API Service**: Provides HTTP endpoints for prediction and recommendation
2. **ML Models**: Two interchangeable models for sentiment analysis:
   - LSTM: A deep learning approach using Long Short-Term Memory neural networks
   - SVM: A machine learning approach using Support Vector Machines with TF-IDF vectorization
3. **Vector Database**: Uses Qdrant for efficient similarity search based on emotional vectors
4. **Reranking System**: Improves recommendation diversity by:
   - Ensuring artist diversity in results
   - Including exploratory recommendations with varying similarity

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service.

### Predict Sentiment
```
POST /predict?save=[bool]
```
Analyzes the sentiment of provided lyrics.
- Input: JSON array of `Track` objects with artist, title, and lyrics
- Output: List of sentiment predictions
- Query parameter `save`: If true, stores the predictions in the database

### Get Similar Songs
```
POST /get-closest?n=[int]
```
Finds songs with similar emotional patterns.
- Input: JSON representation of a `Prediction` object with emotional values
- Output: List of similar tracks
- Query parameter `n`: Number of recommendations to return (default: 1)

## Reranking Algorithm

The system uses a specialized reranking algorithm to ensure diversity in recommendations:
1. Extract the most similar song for each unique artist
2. Include less similar songs for exploration
3. Prioritize diverse recommendations while maintaining similarity

![Reranking Algorithm](RERANKER.md)

## Setup and Deployment

### Prerequisites
- Docker and Docker Compose
- Python 3.12+

### Running with Docker

The service can be deployed using either the LSTM or SVM model:

#### LSTM Version
```
docker build -f Dockerfile.lstm -t lyrics-recommendation-lstm .
docker run -p 5000:5000 lyrics-recommendation-lstm
```

#### SVM Version
```
docker build -f Dockerfile.svm -t lyrics-recommendation-svm .
docker run -p 5000:5000 lyrics-recommendation-svm
```

### Development Setup

1. Install dependencies:
```
uv sync  # For development
uv sync --extra lstm  # For LSTM model development
```

2. Run in debug mode:
```
./run_debug.fish
```

### Populating the Database

To populate the database with initial lyrics data:
```
./populate_db.fish
```

## Dependencies

- Flask: Web framework
- Qdrant: Vector database for similarity search
- NumPy: Numerical computing
- spaCy: NLP preprocessing
- PyTorch: Deep learning for LSTM model
- scikit-learn: Machine learning for SVM model
- Gunicorn: WSGI HTTP Server
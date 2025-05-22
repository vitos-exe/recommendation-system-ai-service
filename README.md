# AI-Powered Lyrics Sentiment Analysis and Recommendation Service

## 1. Overview

This application is an AI-powered service designed to perform sentiment analysis on song lyrics and provide recommendations based on these sentiments. It ingests lyrics, processes them to predict emotional content (angry, happy, relaxed, sad), stores these sentiment vectors in a specialized database, and allows for querying similar songs. The system also incorporates a reranking mechanism to enhance the diversity of recommendations.

The service is built in Python using a microservice-oriented approach, with distinct components for handling data, machine learning, and database interactions. It supports multiple machine learning models for sentiment prediction and utilizes a vector database for efficient similarity searches.

## 2. Features

*   **Sentiment Analysis:** Predicts the emotional sentiment of song lyrics across four categories: angry, happy, relaxed, and sad.
*   **Multiple ML Models:** Supports different models for sentiment analysis:
    *   Long Short-Term Memory (LSTM) network with Word2Vec embeddings.
    *   Term Frequency-Inverse Document Frequency (TF-IDF) vectorization with a Support Vector Machine (SVM) classifier.
*   **Vector Database Storage:** Uses Qdrant to store lyrics and their corresponding sentiment vectors, enabling efficient similarity searches.
*   **Lyrics Recommendation:** Finds songs with similar sentiment profiles.
*   **Reranking:** Implements a reranking strategy to promote artist diversity and include "exploratory" items in the recommendations.
*   **Configurable:** Application behavior can be configured through environment variables (e.g., Qdrant URL, model type, data paths).
*   **Dockerized:** Provides Dockerfiles for deploying the service with either the LSTM or SVM model.
*   **Data Ingestion:** Capable of reading lyrics from a structured folder or a ZIP archive.
*   **Testing Suite:** Includes a comprehensive set of unit tests.

## 3. System Design and Architecture

The application is composed of several key modules:

*   **`ai_service/`**: Main application package.
    *   **`main.py`**: (Currently empty) Intended entry point for the Flask web application, responsible for defining API endpoints.
    *   **`config.py`**: Manages application configuration using environment variables. It defines different configurations for development (`DevConfig`) and testing (`TestConfig`).
    *   **`model.py`**: Defines the core data structures:
        *   `Prediction`: Represents the predicted sentiment scores (angry, happy, relaxed, sad).
        *   `Lyrics`: Represents a song with its artist, title, and predicted sentiment.
        *   `RawLyrics`: Represents a song with its artist, title, and raw lyric text.
    *   **`lyrics_reader.py`**: Handles loading lyrics from the file system. It can read from a directory structure or a ZIP file containing lyrics.
    *   **`ml.py`**: Implements the sentiment analysis logic.
        *   `SentimentModel` (ABC): Defines the interface for sentiment prediction models.
        *   `SentimentLSTM`: Implements sentiment prediction using an LSTM model. It uses pre-trained Word2Vec embeddings (`gensim`) for text vectorization. The LSTM model architecture is defined in `nn_definition.py`.
        *   `SentimentTfidfSvm`: Implements sentiment prediction using TF-IDF vectorization and an SVM classifier (from `scikit-learn`).
        *   `create_model()`: Factory function to instantiate the configured sentiment model.
        *   `get_sentiment_model()`: Retrieves the initialized sentiment model from the Flask application context.
    *   **`nn_definition.py`**: Defines the PyTorch `SentimentLSTMCore` neural network model, including its layers (LSTM, Dropout, Linear) and forward pass logic.
    *   **`db.py`**: Manages interactions with the Qdrant vector database.
        *   Handles client creation and collection setup (`COLLECTION_NAME: "lyrics"`).
        *   Provides functions to convert `Lyrics` objects to Qdrant `PointStruct` and vice-versa.
        *   `add_lyrics()`: Upserts lyrics and their sentiment vectors into Qdrant.
        *   `search_n_closest()`: Queries Qdrant for the `n` most similar lyrics based on a given sentiment prediction vector.
    *   **`reranking.py`**: Implements the logic to rerank a list of candidate lyrics. The current strategy prioritizes artist diversity and includes one "exploratory" item (a less similar item) to broaden recommendations.
    *   **`preprocessing_utils.py`**: (Content not fully shown, but typically contains text preprocessing functions like tokenization, stemming, stop-word removal, etc., used before feeding text to ML models).
*   **`bin/`**: Contains utility scripts.
    *   **`populate_db.fish`**: Script to populate the Qdrant database with lyrics and their predicted sentiments.
    *   **`run_docker.fish`**: Script to build and run the Docker containers for the application.
*   **`tests/`**: Contains unit and integration tests for various components of the application.
    *   Uses `pytest`.
    *   Includes test data (`lyrics_test.zip`) and pre-trained test models (`svc_model.pkl`, `tfidf_vectorizer.pkl`).
*   **`Dockerfile.lstm` & `Dockerfile.svm`**: Dockerfiles to build images for the service, one configured for the LSTM model and the other for the TF-IDF SVM model.
*   **`pyproject.toml`**: Defines project metadata, dependencies, and optional dependencies (for `dev` and `lstm` extras). Uses `uv` for dependency management (implied by `uv.lock`).

### 3.1. Data Flow

1.  **Lyrics Ingestion:** Lyrics are read from a specified path (folder or ZIP file) by `lyrics_reader.py`.
2.  **Sentiment Prediction:**
    *   The raw lyrics are fed into the configured sentiment model (`SentimentLSTM` or `SentimentTfidfSvm` via `ml.py`).
    *   The model preprocesses the text and predicts a 4-dimensional sentiment vector (angry, happy, relaxed, sad).
3.  **Storage:** The lyrics (artist, title) and their corresponding sentiment vectors are stored in the Qdrant database by `db.py`. Each entry gets a unique ID.
4.  **Querying/Recommendation:**
    *   To get recommendations, a target sentiment vector is provided (this could be from a new piece of text or an existing song).
    *   `db.py` queries Qdrant to find the lyrics whose sentiment vectors are closest (using Manhattan distance) to the target vector.
5.  **Reranking:** The initial list of candidates from Qdrant is then passed to `reranking.py`. This module reorders the candidates to ensure artist diversity and potentially introduce an "exploratory" song that is less similar but might be interesting.
6.  **Output:** The reranked list of lyrics is returned to the user/client.

## 4. Technical Stack

*   **Programming Language:** Python (>=3.12.10)
*   **Web Framework:** Flask (implied, though `main.py` is currently empty, other modules use `current_app`)
*   **Machine Learning:**
    *   **LSTM Model:** PyTorch, Gensim (for Word2Vec)
    *   **SVM Model:** Scikit-learn (for TF-IDF and SVM)
    *   **General ML/Data:** NumPy, Pandas, Joblib
*   **Vector Database:** Qdrant
*   **Text Preprocessing:** SpaCy, Pystemmer
*   **Dependency Management:** uv (implied by `uv.lock` and `pyproject.toml`)
*   **Containerization:** Docker
*   **Testing:** Pytest

## 5. Models

The system supports two primary types of sentiment analysis models:

### 5.1. LSTM (Long Short-Term Memory)

*   **Architecture:** Defined in `ai_service/nn_definition.py` (`SentimentLSTMCore`). It consists of an LSTM layer, a dropout layer, and a fully connected linear layer with a softmax activation for outputting probabilities for the four sentiment classes.
*   **Embeddings:** Uses pre-trained Word2Vec embeddings (e.g., `word2vec-google-news-300` from `gensim`) to convert text tokens into dense vectors.
*   **Input:** Expects sequences of word embeddings. `ml.py` handles the conversion of raw lyrics text into these embedding sequences, including padding/truncation to a fixed length.
*   **Output:** A 4-element vector representing the probabilities for (angry, happy, relaxed, sad).
*   **Configuration:** Set `MODEL_TYPE="LSTM"` in the environment or `config.py`.

### 5.2. TF-IDF SVM (Term Frequency-Inverse Document Frequency + Support Vector Machine)

*   **Vectorization:** Uses TF-IDF to convert preprocessed lyrics text into numerical feature vectors. The TF-IDF vectorizer is loaded from a pre-trained file (`tfidf_vectorizer.pkl`).
*   **Classifier:** Uses an SVM classifier (SVC model from `scikit-learn`) to predict sentiment classes based on the TF-IDF vectors. The SVM model is loaded from a pre-trained file (`svc_model.pkl`).
*   **Output:** The SVM's `predict_proba` method is used to get probabilities for each class. If `predict_proba` is not available (e.g., for some SVM configurations), it falls back to `predict` and creates a one-hot encoded probability distribution.
*   **Configuration:** Set `MODEL_TYPE="TFIDF_SVM"` in the environment or `config.py`. This is the default for the test configuration.

## 6. Vector Database (Qdrant)

*   **Purpose:** Stores and indexes lyrics based on their 4-dimensional sentiment vectors.
*   **Collection:** Uses a collection named `"lyrics"`.
*   **Vector Parameters:** The vectors are of size 4, and the distance metric used for similarity search is Manhattan distance (`Distance.MANHATTAN`).
*   **Operations:**
    *   **Setup:** `setup_qdrant` ensures the collection exists when the client is initialized.
    *   **Insertion:** `add_lyrics` converts `Lyrics` objects into `PointStruct` and upserts them into the collection.
    *   **Search:** `search_n_closest` takes a target sentiment vector and queries Qdrant for the most similar points. It fetches an initial pool of candidates (default 5 times the requested `n`) before returning the top `n` (or the reranked list).

## 7. Reranking Logic

The `rerank_lyrics` function in `ai_service/reranking.py` refines the initial list of candidates obtained from Qdrant. The goal is to improve the quality and diversity of recommendations.

The current reranking strategy involves:

1.  **Artist Diversity Pass:** Iterates through the similarity-sorted candidates and picks the most similar song for each new artist encountered. This ensures that the top results aren't dominated by a single artist.
2.  **Guided Exploration:** From the remaining candidates (those not picked for artist diversity), the *least* similar item is selected as an "exploratory item."
3.  **Final List Construction:** The final reranked list is composed of:
    *   The items selected for artist diversity (in their original similarity order relative to each other for that artist).
    *   The single promoted exploratory item.
    *   The rest of the remaining candidates (in their original similarity order).

This approach aims to balance direct relevance with the discovery of new artists and slightly different sentiment profiles.

## 8. Configuration

The application is configured via environment variables, which are accessed through the `Config` class in `ai_service/config.py`. Key configuration options include:

*   `QDRANT_URL`: URL of the Qdrant instance (default: `http://localhost:6333`).
*   `LYRICS_FOLDER_STRUCTURE_PATH`: Path to the lyrics data (folder or ZIP file, default: `lyrics.zip`).
*   `TESTING`: Boolean flag to indicate if the application is in testing mode (default: `False`).
*   `MODELS_DIR`: Directory where trained ML models are stored (default: `models`).
*   `WORD2VEC_NAME`: Name of the Word2Vec model to download via `gensim` (default: `word2vec-google-news-300`), used by the LSTM model.
*   `MODEL_TYPE`: Specifies the sentiment model to use (`LSTM` or `TFIDF_SVM`, default: `LSTM`).

`TestConfig` overrides some of these for the testing environment (e.g., in-memory Qdrant, test data paths, `TFIDF_SVM` model).

## 9. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd recommendation-system-ai-service
    ```

2.  **Install Python:** Ensure Python >= 3.12.10 is installed.

3.  **Install Dependencies:** The project uses `uv` for dependency management.
    ```bash
    pip install uv
    uv pip install -r requirements.txt # Or more commonly:
    uv pip install . # For project dependencies
    uv pip install .[dev] # For development dependencies
    uv pip install .[lstm] # If using the LSTM model and need torch/gensim
    # Or install all:
    uv pip install .[dev,lstm]
    ```
    (Alternatively, if a `requirements.txt` is generated, `uv pip install -r requirements.txt` would be used).
    Given `pyproject.toml`, the standard way is `uv pip install .` and its variants.

4.  **Set up Qdrant:**
    Ensure a Qdrant instance is running and accessible at the URL specified by `QDRANT_URL` (default `http://localhost:6333`). You can run Qdrant using Docker:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```

5.  **Download ML Models (if applicable):**
    *   For **LSTM**: The `SentimentLSTM` model will attempt to download the Word2Vec model specified by `WORD2VEC_NAME` on its first run if not in testing mode. The LSTM model itself (`lstm_sentiment_model.pt`) needs to be pre-trained and placed in the `MODELS_DIR`.
    *   For **TF-IDF SVM**: The `tfidf_vectorizer.pkl` and `svc_model.pkl` files need to be pre-trained and placed in the `MODELS_DIR`. The test versions are in `tests/models/`.

6.  **Prepare Lyrics Data:**
    Place your lyrics data (folder structure or `lyrics.zip`) at the path specified by `LYRICS_FOLDER_STRUCTURE_PATH`. The expected structure inside the zip or folder is:
    ```
    <dataset_name>/
        <label_A>/
            artist1 - title1.txt
            artist2 - title2.txt
        <label_B>/
            artist3 - title3.txt
    ```
    (The `label` folders are iterated but not directly used by the current sentiment prediction which outputs 4 fixed classes. The filename format `artist - title.txt` is parsed.)

## 10. Running the Application

### 10.1. Using Docker

The easiest way to run the application is using Docker, which handles dependencies and model configurations.

1.  **Build the Docker image:**
    *   For LSTM model:
        ```bash
        docker build -t ai-service-lstm -f Dockerfile.lstm .
        ```
    *   For TF-IDF SVM model:
        ```bash
        docker build -t ai-service-svm -f Dockerfile.svm .
        ```

2.  **Run the Docker container:**
    *   For LSTM:
        ```bash
        docker run -p 5000:5000 -e QDRANT_URL="<your_qdrant_url>" -e MODEL_TYPE="LSTM" --name ai-lstm ai-service-lstm
        ```
    *   For SVM:
        ```bash
        docker run -p 5000:5000 -e QDRANT_URL="<your_qdrant_url>" -e MODEL_TYPE="TFIDF_SVM" --name ai-svm ai-service-svm
        ```
    (Adjust port mappings and environment variables as needed. The Flask app, when implemented in `main.py`, would typically run on port 5000).

    The `bin/run_docker.fish` script likely automates parts of this.

### 10.2. Running Locally (Development)

1.  Ensure all dependencies are installed (see Setup).
2.  Set required environment variables (e.g., `QDRANT_URL`, `MODEL_TYPE`, `LYRICS_FOLDER_STRUCTURE_PATH`).
3.  Run the Flask application (once `main.py` is populated):
    ```bash
    flask --app ai_service.main run --debug
    ```
    (This command assumes `main.py` will define a Flask `app` object).

## 11. Populating the Database

The `bin/populate_db.fish` script is provided to ingest lyrics, perform sentiment analysis, and store the results in Qdrant. This script would typically:

1.  Initialize the application context.
2.  Read lyrics using `lyrics_reader.py`.
3.  For each lyric, get sentiment predictions using `ml.py`.
4.  Store the lyrics and predictions in Qdrant using `db.py`.

Execute it using Fish shell:
```bash
fish bin/populate_db.fish
```
Ensure environment variables are correctly set for the script's execution context.

## 12. Testing

The project includes a suite of tests in the `tests/` directory.

*   **Run tests using `pytest`:**
    ```bash
    pytest
    ```
    This will discover and run all tests in the `tests` directory. The tests cover individual modules like database interaction (`test_db.py`), model predictions (`test_ml.py`), data models (`test_model.py`), and reranking logic (`test_reranking.py`).
    The test configuration (`TestConfig`) uses in-memory Qdrant and specific test data/models.

This README should provide a good overview of your application.

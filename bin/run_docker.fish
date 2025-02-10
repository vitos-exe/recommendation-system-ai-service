# Expects image with tag "ai-service" to be built
# Run from project root folder
# Ensure you have lyrics and model folders
# Optionally you can exclude gensim-data volume
docker run \
--rm \
-p 5000:5000 \
-v "$HOME/gensim-data:/root/gensim-data" \
-v "$PWD/lyrics:/app/lyrics" \
-v "$PWD/models:/app/models" \
--name=ai-service \
--network="host" \
ai-service

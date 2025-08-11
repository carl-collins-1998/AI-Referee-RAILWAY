#!/bin/bash

# Print environment info
echo "Starting Basketball Referee API on Railway"
echo "PORT: $PORT"
echo "MODEL_URL: ${MODEL_URL:0:50}..."
echo "Working directory: $(pwd)"
echo "Contents: $(ls -la)"

# Create models directory
mkdir -p models

# Download model if URL is set and file doesn't exist
if [ ! -f "models/best.pt" ] && [ -n "$MODEL_URL" ]; then
    echo "Downloading model..."
    wget -O models/best.pt "$MODEL_URL" || echo "Model download failed"
fi

# Start the app
python main.py
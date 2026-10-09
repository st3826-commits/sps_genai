# FastAPI Text Generation and Word Embeddings

This project extends the class FastAPI application with a spaCy word embedding endpoint.

## Setup

Install dependencies and download the English vector model:

```bash
uv sync
uv run python -m spacy download en_core_web_md
```

```bash
uv run fastapi dev app/main.py
```

Interactive API documentation:
http://127.0.0.1:8000/docs

## Endpoints

- GET /: Returns a greeting.
- POST /generate: Generates text using the bigram model.
- GET /embedding?word=love: Returns a word and its embedding.
- POST /classify: Classifies an uploaded image using the trained CNN.

The embedding endpoint uses spaCy's en_core_web_md model.
A successful response contains:
- word: The queried word.
- dimension: The vector length (300).
- embedding: The complete list of vector values.

## Input Validation
- Empty or whitespace-only input: HTTP 400.
- Input containing multiple tokens: HTTP 400.
- A token without a vector: HTTP 404.
- Missing word parameter: HTTP 422.

## Manual Checks 
- love: Returned a 300-dimensional embedding.
- Empty input: Returned "Please enter a word."
- hello world: Returned "Please enter a single word."


# With Docker
Docker must be installed and running.

From the project root, build the image:

```bash
docker build -t sps-genai .
```

Start the container:

```bash
docker run --rm -p 8001:8000 sps-genai
```

The image includes the Python dependencies and spaCy's
en_core_web_md model. Internet access is required during the build.

Test the embedding endpoint:
http://127.0.0.1:8001/embedding?word=love

Interactive API documentation:
http://127.0.0.1:8001/docs

A successful embedding response contains the queried word,
dimension 300, and the complete embedding list.

Press Ctrl+C in the terminal to stop the container.

# CNN Image Classification
## Model architecture

Input: RGB images resized to 64 × 64.
- Conv2d: 3 → 16 channels, kernel 3, stride 1, padding 1 → ReLU → MaxPool2d(2, 2).
- Conv2d: 16 → 32 channels, kernel 3, stride 1, padding 1 → ReLU → MaxPool2d(2, 2).
- Flatten → Linear(8192, 100) → ReLU → Linear(100, 10).

Training and inference use identical preprocessing: resize to 64 × 64, convert to a tensor, and normalize each channel with mean and standard deviation 0.5.

## Training
From the project root, run:

uv sync
uv run python -m app.train_cnn

Uses 50,000 training images and 10,000 test images, with 5 epochs, batch size 64, Adam (learning rate 0.001), and CrossEntropyLoss.
Recorded results: 74.78% final epoch training accuracy; 64.97% test accuracy. Results may vary when retraining.
Weights: models/cifar10_cnn.pt
Metrics: models/training_metrics.json

## API

POST /classify accepts an image in the multipart field file:
curl -X POST "http://127.0.0.1:8001/classify" \
  -F "file=@example.jpg"

Example response:

{
  "class": "bird",
  "class_id": 2,
  "confidence": 0.5432
}

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Confidence is a softmax score, not a guarantee of correctness.
Responses: **200** for successful prediction, **400** for an invalid image, **422** for a missing file. Prediction was verified in Docker.

## Files

- app/cnn_model.py: architecture.
- app/image_data.py: preprocessing and data loading.
- app/train_cnn.py: training and evaluation.
- app/cnn_api.py: prediction endpoint.
- app/main.py: FastAPI application.
- models/: weights and metrics.

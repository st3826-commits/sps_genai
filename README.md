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
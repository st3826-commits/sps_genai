from pathlib import Path
import torch
from app.lstm_model import LSTMModel
from app.lstm_model import generate_with_rnn as generate_rnn_text

from fastapi import FastAPI, HTTPException
import spacy
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.cnn_api import router as cnn_router
from app.gan_api import router as gan_router





app = FastAPI()
app.include_router(gan_router)
app.include_router(cnn_router)
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]


bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {"generated_text": generated_text}

@app.get("/embedding")
def get_embedding(word: str):
    # Remove spaces before and after the input
    word = word.strip()

    if not word:
        raise HTTPException(
            status_code=400,
            detail="Please enter a word."
        )

    doc = nlp(word)

    if len(doc) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please enter a single word."
        )

    token = doc[0]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail="No embedding found for this word."
        )

    return {
        "word": token.text,
        "dimension": len(token.vector),
        "embedding": token.vector.tolist()
    }

@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    if not 1 <= request.length <= 1000:
        raise HTTPException(
            status_code=400,
            detail="length must be between 1 and 1000"
        )

    seed_text = request.start_word.strip().lower()

    if not seed_text or len(seed_text) > 200:
        raise HTTPException(
            status_code=400,
            detail="start_word must contain 1 to 200 characters"
        )

    try:
        generated_text = generate_rnn_text(
            model=rnn_model,
            seed_text=seed_text,
            stoi=rnn_stoi,
            itos=rnn_itos,
            length=request.length,
            temperature=0.8
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    return {"generated_text": generated_text}



checkpoint_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "module7_lstm.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

rnn_chars = checkpoint["chars"]
rnn_stoi = {
    char: index for index, char in enumerate(rnn_chars)
}
rnn_itos = {
    index: char for index, char in enumerate(rnn_chars)
}

rnn_model = LSTMModel(
    vocab_size=len(rnn_chars),
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"]
)

rnn_model.load_state_dict(checkpoint["model_state_dict"])
rnn_model.eval()


import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        logits = self.fc(x)

        return logits, hidden




def generate_with_rnn(
    model,
    seed_text,
    stoi,
    itos,
    length=100,
    temperature=1.0
):
    if not seed_text:
        raise ValueError("seed_text must not be empty")
    if length < 0:
        raise ValueError("length must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    unknown = sorted(set(seed_text) - set(stoi))
    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[stoi[char] for char in seed_text]],
        dtype=torch.long,
        device=device
    )

    generated = list(seed_text)
    hidden = None
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(length):
                logits, hidden = model(input_ids, hidden)
                logits = logits[:, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=-1)

                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1
                ).item()

                generated.append(itos[next_id])

                input_ids = torch.tensor(
                    [[next_id]],
                    dtype=torch.long,
                    device=device
                )
    finally:
        model.train(was_training)

    return "".join(generated)
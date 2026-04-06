"""
CNN-BiLSTM-Attention Hybrid Model for Sentiment Analysis — PyTorch Implementation.

Architecture:
  Input → Embedding → [Conv1D(k=3) | Conv1D(k=5) | Conv1D(k=7)]
  → Concatenate (preserve sequence) → BiLSTM → Self-Attention → Dense → Output

Designed for GPU/CPU training on a 250K mixed IMDB+Yelp corpus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Default Hyperparameters ────────────────────────────────
DEFAULT_VOCAB_SIZE = 20000
DEFAULT_EMBED_DIM = 128
DEFAULT_MAX_LEN = 200
DEFAULT_CNN_FILTERS = 32   # Reduced from 64 — let BiLSTM+Attention do the heavy logic
DEFAULT_LSTM_UNITS = 64
DEFAULT_DROPOUT = 0.6      # Increased from 0.5 — forces reliance on sequence, not keywords
DEFAULT_NUM_CLASSES = 1    # binary (sigmoid); set >1 for multi-class (softmax)


# ─── Self-Attention Layer ───────────────────────────────────
class SelfAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over BiLSTM outputs.
    Learns to weight every timestep so the model can look at 'not'
    and 'bad' simultaneously, no matter how far apart they are.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden_dim)
        scores = self.attention(lstm_outputs)       # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)          # (batch, seq_len, 1)
        # Weighted sum → context vector of the whole sentence
        context = torch.sum(weights * lstm_outputs, dim=1)  # (batch, hidden_dim)
        return context, weights


class CNNBiLSTMSentiment(nn.Module):
    """
    CNN-BiLSTM-Attention hybrid model for text classification.

    CNN branches extract local n-gram features (3/5/7-grams).
    BiLSTM captures sequential context from the full preserved sequence.
    Self-Attention learns which timesteps matter most for the final decision.
    """

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        cnn_filters: int = DEFAULT_CNN_FILTERS,
        kernel_sizes: tuple = (3, 5, 7),
        lstm_units: int = DEFAULT_LSTM_UNITS,
        dropout_rate: float = DEFAULT_DROPOUT,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.cnn_filters = cnn_filters
        self.num_classes = num_classes

        # ── Embedding ────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ── Parallel CNN Branches ────────────────────────────
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=cnn_filters,
                kernel_size=ks,
                padding=ks // 2,  # preserves sequence length
            )
            for ks in kernel_sizes
        ])

        # ── Batch Norm + Dropout after CNN ───────────────────
        total_cnn_out = cnn_filters * len(kernel_sizes)
        self.bn_cnn = nn.BatchNorm1d(total_cnn_out)
        self.dropout_cnn = nn.Dropout(dropout_rate)

        # ── BiLSTM (2-layer for deeper sequence comprehension) ──
        self.bilstm = nn.LSTM(
            input_size=total_cnn_out,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # ── Self-Attention ───────────────────────────────────
        self.attention = SelfAttention(hidden_dim=lstm_units * 2)

        # ── Dense Head ───────────────────────────────────────
        self.dense = nn.Sequential(
            nn.Linear(lstm_units * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
        )

        # ── Output ───────────────────────────────────────────
        self.output_layer = nn.Linear(64, num_classes if num_classes > 1 else 1)

    def forward(self, x):
        # x: (batch, seq_len) integer tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # Conv1D expects (batch, channels, seq_len)
        embedded_t = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Parallel convolutions — NO pooling, preserve sequence length
        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(embedded_t))  # (batch, filters, seq_len)
            conv_outputs.append(c)

        # Concatenate along filter dim: (batch, filters * num_kernels, seq_len)
        concat = torch.cat(conv_outputs, dim=1)
        concat = self.bn_cnn(concat)
        concat = self.dropout_cnn(concat)

        # Transpose for BiLSTM: (batch, seq_len, total_filters)
        lstm_input = concat.transpose(1, 2)

        # BiLSTM reads the full sequence
        lstm_out, _ = self.bilstm(lstm_input)  # (batch, seq_len, lstm*2)

        # Self-Attention: weight every timestep instead of just taking the last one
        context_vector, attn_weights = self.attention(lstm_out)

        # Dense
        x = self.dense(context_vector)

        # Output
        logits = self.output_layer(x)

        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        else:
            return logits  # apply softmax in loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model_summary_dict(model: CNNBiLSTMSentiment) -> dict:
    """Extract model architecture info as a dictionary for the API."""
    layers_info = []

    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if name == "":
            continue  # skip root
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params == 0 and not isinstance(module, (nn.ModuleList, nn.Sequential)):
            # Skip wrappers with no direct params except for meaningful layers
            if not isinstance(module, (nn.ReLU, nn.Dropout, nn.BatchNorm1d)):
                continue

        layer_info = {
            "name": name,
            "type": module.__class__.__name__,
            "output_shape": "—",
            "params": params,
        }
        layers_info.append(layer_info)

    return {
        "model_name": "CNN_BiLSTM_Attention_Sentiment",
        "total_params": model.count_parameters(),
        "trainable_params": model.count_trainable_parameters(),
        "layers": layers_info,
        "architecture": "CNN (multi-kernel) → BiLSTM → Self-Attention → Dense",
    }


if __name__ == "__main__":
    m = CNNBiLSTMSentiment()
    print(m)
    print(f"\nTotal params: {m.count_parameters():,}")
    print(f"Trainable params: {m.count_trainable_parameters():,}")

    # Quick forward pass test
    dummy = torch.randint(0, 20000, (2, 200))
    out = m(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")

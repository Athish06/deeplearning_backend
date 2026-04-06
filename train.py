"""
Training script for the CNN-BiLSTM Sentiment Analysis model — PyTorch.

Usage:
    cd d:\\Projects\\DL\\backend
    python train.py

Downloads IMDB (~84MB) + Yelp (~160MB) datasets and trains on a
mixed 250K-row corpus. GPU recommended (~90-120s per epoch).
"""

import os
import sys
import json
import time
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

from model import CNNBiLSTMSentiment
from sarcasm_augmentation import get_sarcasm_examples
from preprocessing import (
    VOCAB_SIZE,
    MAX_LEN,
    SimpleTokenizer,
    clean_text,
    pad_sequences,
    save_tokenizer,
)

# ─── Configuration ───────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "sentiment_model.pt")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.pkl")
METRICS_PATH = os.path.join(SAVE_DIR, "training_metrics.json")

BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_imdb():
    """
    Download and parse the IMDB dataset.
    Uses the IMDB dataset from Andrew Maas (Stanford), stored as text files.
    Falls back to a simple built-in approach.
    """
    import urllib.request
    import tarfile
    import shutil

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = os.path.join(SAVE_DIR, "aclImdb_v1.tar.gz")
    extract_dir = os.path.join(SAVE_DIR, "aclImdb")

    if os.path.exists(extract_dir):
        print("  → IMDB dataset already downloaded.")
    else:
        print(f"  → Downloading IMDB dataset (~84MB)...")
        os.makedirs(SAVE_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, tar_path)
        print(f"  → Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(SAVE_DIR)
        os.remove(tar_path)

    # Parse
    texts = []
    labels = []
    for split in ["train", "test"]:
        for label_name, label_val in [("pos", 1), ("neg", 0)]:
            folder = os.path.join(extract_dir, split, label_name)
            for fname in os.listdir(folder):
                if fname.endswith(".txt"):
                    with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(label_val)

    return texts, labels


def get_mixed_dataset():
    """Download and merge 50K IMDB + 200K Yelp rows into a balanced 250K dataset."""
    print("\n📦 Loading and mixing datasets...")

    # 1. Load the 50k IMDB rows
    imdb_texts, imdb_labels = download_imdb()
    imdb_df = pd.DataFrame({'text': imdb_texts, 'label': imdb_labels})

    # 2. Download Yelp (if not already downloaded)
    yelp_url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz"
    yelp_tar_path = os.path.join(SAVE_DIR, "yelp_review_polarity_csv.tgz")
    yelp_csv_path = os.path.join(SAVE_DIR, "yelp_review_polarity_csv", "yelp_review_polarity_csv", "train.csv")

    if not os.path.exists(yelp_csv_path):
        print("  → Downloading Yelp dataset (~160MB)...")
        urllib.request.urlretrieve(yelp_url, yelp_tar_path)
        print("  → Extracting Yelp...")
        with tarfile.open(yelp_tar_path, "r:gz") as tar:
            tar.extractall(SAVE_DIR)
        os.remove(yelp_tar_path)

    # 3. Parse Yelp
    print("  → Parsing Yelp...")
    yelp_df = pd.read_csv(yelp_csv_path, header=None, names=['label', 'text'])
    yelp_df['label'] = yelp_df['label'].map({1: 0, 2: 1})  # Map to 0 (Neg) and 1 (Pos)

    # 4. Sample exactly 200,000 Yelp rows (100k Neg, 100k Pos)
    print("  → Sampling 200,000 Yelp rows...")
    yelp_sampled = pd.concat([
        yelp_df[yelp_df['label'] == 0].sample(100000, random_state=42),
        yelp_df[yelp_df['label'] == 1].sample(100000, random_state=42)
    ])

    # 5. Combine and Shuffle the full 250k dataset
    combined_df = pd.concat([imdb_df, yelp_sampled])
    combined_df = sk_shuffle(combined_df, random_state=42).reset_index(drop=True)

    print(f"  → Total combined samples ready for training: {len(combined_df)}")
    return combined_df['text'].tolist(), combined_df['label'].tolist()


def train():
    """Main training loop."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 60)
    print("CNN-BiLSTM-Attention Sentiment Analysis — Training")
    print("=" * 60)

    # ── Load Data ────────────────────────────────────────────
    print("\n📦 Loading Mixed dataset...")
    texts, labels = get_mixed_dataset()

    # Inject sarcasm augmentation data
    sarc_texts, sarc_labels = get_sarcasm_examples()
    texts.extend(sarc_texts)
    labels.extend(sarc_labels)

    print(f"  → Total samples (with sarcasm augmentation): {len(texts)}")
    print(f"  → Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

    # ── Split ────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"  → Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    # ── Clean ────────────────────────────────────────────────
    print("\n🧹 Cleaning text...")
    X_train_clean = [clean_text(t) for t in X_train_raw]
    X_val_clean = [clean_text(t) for t in X_val_raw]
    X_test_clean = [clean_text(t) for t in X_test_raw]

    # ── Tokenize ─────────────────────────────────────────────
    print("🔤 Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(X_train_clean)
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    print(f"  → Vocabulary size: {tokenizer.actual_vocab_size}")

    # ── Pad Sequences ────────────────────────────────────────
    print("📏 Padding sequences...")
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_clean), MAX_LEN)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_clean), MAX_LEN)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_clean), MAX_LEN)

    # ── Create DataLoaders ───────────────────────────────────
    def make_loader(X, y, shuffle=False):
        X_t = torch.LongTensor(X)
        y_t = torch.FloatTensor(y)
        ds = TensorDataset(X_t, y_t)
        # pin_memory ensures fast transfer to the GPU
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2, pin_memory=True)

    train_loader = make_loader(X_train_seq, y_train, shuffle=True)
    val_loader = make_loader(X_val_seq, y_val)
    test_loader = make_loader(X_test_seq, y_test)

    # ── Build Model ──────────────────────────────────────────
    print("\n🏗️  Building CNN-BiLSTM model...")
    model = CNNBiLSTMSentiment(
        vocab_size=VOCAB_SIZE,
        embed_dim=128,
        cnn_filters=32,
        kernel_sizes=(3, 5, 7),
        lstm_units=64,
        dropout_rate=0.6,
        num_classes=1,
    ).to(DEVICE)

    print(model)
    print(f"  → Total params: {model.count_parameters():,}")
    print(f"  → Trainable params: {model.count_trainable_parameters():,}")

    # ── Loss & Optimizer ─────────────────────────────────────
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # ── Training Loop ────────────────────────────────────────
    print("\n🚀 Training started...")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            preds = (outputs >= 0.5).long()
            train_correct += (preds == batch_y.long()).sum().item()
            train_total += batch_X.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).long()
                val_correct += (preds == batch_y.long()).sum().item()
                val_total += batch_X.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        elapsed = time.time() - start_time
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Step scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── Load Best Model ──────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    # ── Evaluate on Test Set ─────────────────────────────────
    print("\n📊 Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            test_total += batch_X.size(0)
            preds = (outputs >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.long().cpu().numpy())

    test_loss /= test_total
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print(f"  → Test Loss: {test_loss:.4f}")
    print(f"  → Test Accuracy: {test_acc:.4f}")

    print("\n📋 Classification Report:")
    report = classification_report(
        all_labels, all_preds,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(all_labels, all_preds)
    print("🔲 Confusion Matrix:")
    print(cm)

    # ── Save Metrics ─────────────────────────────────────────
    metrics = {
        "trained_at": datetime.now().isoformat(),
        "dataset": "IMDB + Yelp + Sarcasm (300K)",
        "vocab_size": VOCAB_SIZE,
        "max_sequence_length": MAX_LEN,
        "embedding_dim": 128,
        "epochs_trained": len(history["accuracy"]),
        "batch_size": BATCH_SIZE,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
        "labels": ["Negative", "Positive"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Tokenizer saved to: {TOKENIZER_PATH}")
    print(f"✅ Metrics saved to: {METRICS_PATH}")
    print(f"\n🎯 Final Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    train()

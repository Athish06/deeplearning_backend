"""
Prediction pipeline for sentiment analysis using Hugging Face SiEBERT.
Keeps API-compatible output while moving runtime inference away from the local CNN-BiLSTM model.
"""

import os
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import clean_text

# ─── Paths ───────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
METRICS_PATH = os.path.join(SAVE_DIR, "training_metrics.json")
HF_MODEL_NAME = "siebert/sentiment-roberta-large-english"

LABELS = ["Negative", "Positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentPredictor:
    """
    Encapsulates model loading and prediction.
    Instantiate once at server startup; call predict() per request.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.metrics = None
        self.id2label = {}
        self._loaded = False

    def load(self) -> bool:
        """Load Hugging Face model/tokenizer and optional legacy metrics. Returns True on success."""
        try:
            print(f"🔁 Loading Hugging Face tokenizer: {HF_MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

            print(f"🔁 Loading Hugging Face model: {HF_MODEL_NAME}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
            self.model.to(DEVICE)
            self.model.eval()

            self.id2label = {
                int(k): v
                for k, v in self.model.config.id2label.items()
            }

            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    self.metrics = json.load(f)

            self._loaded = True
            print("✅ Hugging Face model loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single text input.

        Returns:
            {
                "label": "Positive" | "Negative",
                "confidence": 0.0-1.0,
                "probabilities": {"Negative": 0.xx, "Positive": 0.xx},
                "input_length": int,
                "cleaned_text_preview": str (first 200 chars)
            }
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Keep legacy cleaning only for preview/debugging, use raw text for transformer inference.
        cleaned = clean_text(text)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        def _norm(label: str) -> str:
            return label.lower().replace("_", "").replace("-", "")

        neg_idx = 0
        pos_idx = 1 if len(probs) > 1 else 0
        for idx, label in self.id2label.items():
            normalized = _norm(label)
            if "neg" in normalized:
                neg_idx = idx
            elif "pos" in normalized:
                pos_idx = idx

        prob_negative = float(probs[neg_idx])
        prob_positive = float(probs[pos_idx])

        label = "Positive" if prob_positive >= 0.5 else "Negative"
        confidence = max(prob_positive, prob_negative)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Negative": round(prob_negative, 4),
                "Positive": round(prob_positive, 4),
            },
            "input_length": len(text.split()),
            "cleaned_text_preview": cleaned[:200],
        }

    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for a batch of texts."""
        return [self.predict(t) for t in texts]

    def get_model_info(self) -> dict:
        """Return model architecture + training metrics for the API."""
        if not self._loaded:
            return {"error": "Model not loaded"}

        info = {
            "model_name": "SiEBERT_RoBERTa_Large",
            "provider": "Hugging Face",
            "model_id": HF_MODEL_NAME,
            "architecture": self.model.__class__.__name__,
            "labels": self.id2label,
            "device": str(DEVICE),
            "num_parameters": int(self.model.num_parameters()),
        }

        if self.metrics:
            info["legacy_cnn_bilstm_metrics"] = {
                "dataset": self.metrics.get("dataset", "Unknown"),
                "trained_at": self.metrics.get("trained_at", "Unknown"),
                "test_accuracy": self.metrics.get("test_accuracy", 0),
                "test_loss": self.metrics.get("test_loss", 0),
                "epochs_trained": self.metrics.get("epochs_trained", 0),
                "batch_size": self.metrics.get("batch_size", 0),
                "vocab_size": self.metrics.get("vocab_size", 0),
                "max_sequence_length": self.metrics.get("max_sequence_length", 0),
                "embedding_dim": self.metrics.get("embedding_dim", 0),
                "labels": self.metrics.get("labels", []),
                "training_history": self.metrics.get("training_history", {}),
                "classification_report": self.metrics.get("classification_report", {}),
                "confusion_matrix": self.metrics.get("confusion_matrix", []),
            }
        return info

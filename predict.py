"""
Prediction pipeline for sentiment analysis using SiEBERT.

Supported backends:
1) local  - runs the model in-process (higher memory)
2) hf_api - calls the hosted Hugging Face inference API (low memory)
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from preprocessing import clean_text

# Paths
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
METRICS_PATH = os.path.join(SAVE_DIR, "training_metrics.json")

# Model/inference constants
HF_MODEL_NAME = "siebert/sentiment-roberta-large-english"
HF_ROUTER_BASE = "https://router.huggingface.co"
HF_LEGACY_API_BASE = "https://api-inference.huggingface.co"
HF_API_URL_DEFAULT = f"{HF_ROUTER_BASE}/hf-inference/models/{HF_MODEL_NAME}"

LABELS = ["Negative", "Positive"]
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in TRUTHY_VALUES


def _normalize_label(label: str) -> str:
    return label.lower().replace("_", "").replace("-", "").replace(" ", "")


def _normalize_hf_api_url(url: str) -> str:
    """Normalize user-provided URL and transparently migrate legacy API host."""
    normalized = (url or "").strip()
    if not normalized:
        return HF_API_URL_DEFAULT

    if normalized.startswith(HF_LEGACY_API_BASE):
        suffix = normalized[len(HF_LEGACY_API_BASE):]
        if not suffix.startswith("/"):
            suffix = f"/{suffix}"

        # Legacy endpoints used /models/{repo}; router now expects /hf-inference/models/{repo}.
        if suffix.startswith("/models/"):
            return f"{HF_ROUTER_BASE}/hf-inference{suffix}"

        return f"{HF_ROUTER_BASE}{suffix}"

    return normalized


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
        self._torch = None
        self.device = "cpu"

        backend = os.getenv("SENTIMENT_BACKEND", "").strip().lower()
        if backend:
            self.backend = backend
        else:
            # Default to hosted API on Render to avoid memory OOM.
            self.backend = "hf_api" if os.getenv("RENDER") else "local"

        self.hf_api_url = _normalize_hf_api_url(os.getenv("HF_API_URL", HF_API_URL_DEFAULT))
        self.hf_api_token = os.getenv("HF_API_TOKEN", "").strip()
        self.hf_api_timeout = int(os.getenv("HF_API_TIMEOUT", "60"))

    def _load_metrics(self) -> None:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def _load_local_model(self, auto_model_cls):
        # low_cpu_mem_usage keeps peak memory lower during model load.
        try:
            return auto_model_cls.from_pretrained(HF_MODEL_NAME, low_cpu_mem_usage=True)
        except TypeError:
            return auto_model_cls.from_pretrained(HF_MODEL_NAME)
        except Exception as exc:
            text = str(exc).lower()
            if "low_cpu_mem_usage" in text or "accelerate" in text:
                return auto_model_cls.from_pretrained(HF_MODEL_NAME)
            raise

    def _load_local(self) -> bool:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"INFO: Loading tokenizer from {HF_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

        print(f"INFO: Loading model weights from {HF_MODEL_NAME}...")
        model = self._load_local_model(AutoModelForSequenceClassification)

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self._loaded = True

        print(f"INFO: Local inference ready on device={self.device}.")
        return True

    def _load_hf_api(self) -> bool:
        if not self.hf_api_token:
            print("WARN: HF_API_TOKEN not set. Inference API may be rate-limited.")

        self.model = None
        self.tokenizer = None
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self._loaded = True
        print(f"INFO: Hosted inference backend enabled for model={HF_MODEL_NAME}.")
        return True

    def load(self) -> bool:
        """Load predictor backend and optional legacy metrics. Returns True on success."""
        try:
            self._load_metrics()

            if self.backend == "hf_api":
                return self._load_hf_api()

            if self.backend == "local":
                try:
                    return self._load_local()
                except Exception as local_error:
                    allow_fallback = _is_truthy(os.getenv("ALLOW_BACKEND_FALLBACK", "true"))
                    print(f"ERROR: Local backend failed: {local_error}")
                    if allow_fallback:
                        print("INFO: Falling back to hosted hf_api backend.")
                        self.backend = "hf_api"
                        return self._load_hf_api()
                    self._loaded = False
                    return False

            print(f"ERROR: Unsupported backend '{self.backend}'. Use 'local' or 'hf_api'.")
            self._loaded = False
            return False

        except Exception as exc:
            print(f"ERROR: Failed to load predictor: {exc}")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _build_response(self, text: str, cleaned: str, prob_negative: float, prob_positive: float) -> dict:
        # Keep response schema unchanged for frontend/API compatibility.
        total = prob_negative + prob_positive
        if total > 0:
            prob_negative /= total
            prob_positive /= total

        label = "Positive" if prob_positive >= 0.5 else "Negative"
        confidence = max(prob_positive, prob_negative)

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                "Negative": round(float(prob_negative), 4),
                "Positive": round(float(prob_positive), 4),
            },
            "input_length": len(text.split()),
            "cleaned_text_preview": cleaned[:200],
        }

    def _predict_local(self, text: str, cleaned: str) -> dict:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            logits = self.model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        neg_idx = 0
        pos_idx = 1 if len(probs) > 1 else 0
        for idx, label in self.id2label.items():
            normalized = _normalize_label(label)
            if "neg" in normalized or normalized.endswith("0"):
                neg_idx = idx
            elif "pos" in normalized or normalized.endswith("1"):
                pos_idx = idx

        prob_negative = float(probs[neg_idx])
        prob_positive = float(probs[pos_idx])
        return self._build_response(text, cleaned, prob_negative, prob_positive)

    def _request_hf_api(self, text: str) -> Any:
        payload = {
            "inputs": text,
            "parameters": {
                "top_k": 2,
                "return_all_scores": True,
            },
            "options": {
                "wait_for_model": True,
            },
        }

        headers = {"Content-Type": "application/json"}
        if self.hf_api_token:
            headers["Authorization"] = f"Bearer {self.hf_api_token}"

        request = urllib.request.Request(
            self.hf_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.hf_api_timeout) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HF API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HF API network error: {exc.reason}") from exc

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HF API returned non-JSON response: {response_text[:200]}") from exc

        if isinstance(parsed, dict) and "error" in parsed:
            raise RuntimeError(str(parsed["error"]))

        return parsed

    def _predict_hf_api(self, text: str, cleaned: str) -> dict:
        payload = self._request_hf_api(text)

        items = []
        if isinstance(payload, list):
            if payload and isinstance(payload[0], list):
                items = payload[0]
            else:
                items = payload
        elif isinstance(payload, dict) and {"label", "score"}.issubset(payload.keys()):
            items = [payload]

        if not items:
            raise RuntimeError("HF API response did not include label scores.")

        prob_negative = None
        prob_positive = None
        for item in items:
            if not isinstance(item, dict):
                continue

            label = _normalize_label(str(item.get("label", "")))
            score = float(item.get("score", 0.0))

            if "neg" in label or label.endswith("0"):
                prob_negative = score
            elif "pos" in label or label.endswith("1"):
                prob_positive = score

        if prob_negative is None and prob_positive is None:
            # Last-resort fallback for unusual label names.
            if len(items) >= 2 and all(isinstance(i, dict) and "score" in i for i in items[:2]):
                prob_negative = float(items[0]["score"])
                prob_positive = float(items[1]["score"])
            else:
                raise RuntimeError("Could not map HF API labels to negative/positive classes.")

        if prob_negative is None:
            prob_negative = max(0.0, min(1.0, 1.0 - prob_positive))
        if prob_positive is None:
            prob_positive = max(0.0, min(1.0, 1.0 - prob_negative))

        return self._build_response(text, cleaned, prob_negative, prob_positive)

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
            raise RuntimeError("Model backend not loaded. Call load() first.")

        cleaned = clean_text(text)

        if self.backend == "local":
            return self._predict_local(text, cleaned)
        if self.backend == "hf_api":
            return self._predict_hf_api(text, cleaned)

        raise RuntimeError(f"Unsupported backend '{self.backend}'.")

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
            "inference_backend": self.backend,
            "labels": self.id2label,
        }

        if self.backend == "local" and self.model is not None:
            info.update(
                {
                    "architecture": self.model.__class__.__name__,
                    "device": str(self.device),
                    "num_parameters": int(self.model.num_parameters()),
                }
            )
        else:
            info.update(
                {
                    "architecture": "HostedInferenceAPI",
                    "device": "remote",
                }
            )

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

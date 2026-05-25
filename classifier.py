import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentClassifier:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        # Mapping matching tweet_eval data settings
        self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1).squeeze().tolist()

        pred_label_id = torch.argmax(outputs.logits, dim=1).item()

        return {
            "sentiment": self.label_mapping[pred_label_id],
            "confidence": round(probabilities[pred_label_id], 4),
            "probabilities": {
                "negative": round(probabilities[0], 4),
                "neutral": round(probabilities[1], 4),
                "positive": round(probabilities[2], 4)
            }
        }
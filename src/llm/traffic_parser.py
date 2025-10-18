import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class LogLLMEmbedder:
    """
    Uses a lightweight transformer to embed textual network log lines.
    Designed to produce semantic enrichment features for downstream GNN.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 64, device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding logs"):
                batch = texts[i:i+batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**encoded)
                # Use CLS token or mean pooling
                last_hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                mean_pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embeddings.append(mean_pooled.cpu())
        return torch.cat(embeddings, dim=0)

def load_logs(log_path: str) -> List[str]:
    logs = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                logs.append(obj.get("raw", line))
            except json.JSONDecodeError:
                logs.append(line)
    return logs

def build_semantic_feature_table(log_path: str, output_path: str, model_name: str, max_length: int, batch_size: int = 16):
    logs = load_logs(log_path)
    embedder = LogLLMEmbedder(model_name=model_name, max_length=max_length)
    embeddings = embedder.embed_texts(logs, batch_size=batch_size)
    torch.save({"embeddings": embeddings, "lines": logs}, output_path)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="data/processed/logs.jsonl")
    parser.add_argument("--out", type=str, default="data/processed/log_embeddings.pt")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    build_semantic_feature_table(args.logs, args.out, args.model_name, args.max_length, args.batch_size)
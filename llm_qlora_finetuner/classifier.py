"""Fine-tune a transformer encoder for text classification with LoRA (optionally QLoRA).

The base encoder (any Hugging Face ``AutoModel``, e.g. a small multilingual sentence encoder) is
frozen; its attention projections get :class:`~llm_qlora_finetuner.finetuner.LoraLinear` adapters
from this package — with the frozen weights stored in NF4 when ``quantize_4bit`` — and a linear
head over mean-pooled token states is trained together with the adapters. Only the adapters and
the head are saved (kilobytes to a few megabytes); the base model is re-downloaded by name.

Context tokens (``["active_project"]``) are prepended to the text as ``[active_project]``, so a
classifier can learn that the same words route differently in a different situation.

The saved directory has ``model.json`` with ``"kind": "lora-classifier"``, the format other
components (a model registry, a router) can load through a loader the integrator registers.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .finetuner import LoraConfig, QLoraPipeline

KIND = "lora-classifier"
Context = Optional[Iterable[str]]


def with_context(text: str, context: Context) -> str:
    tags = " ".join(f"[{c}]" for c in (context or ()))
    return f"{tags} {text}".strip()


class _Model(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, n_labels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)


class LoraClassifier:
    """Train, predict, save and load a LoRA-adapted encoder classifier."""

    kind = KIND

    def __init__(
        self,
        labels: Sequence[str],
        base_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        lora: Optional[LoraConfig] = None,
        quantize_4bit: bool = True,
        max_length: int = 48,
        threads: Optional[int] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.labels = list(labels)
        self.base_model = base_model
        self.lora = lora or LoraConfig(r=8, alpha=16, target_modules=["query", "value"], dropout=0.05)
        self.quantize_4bit = quantize_4bit
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        encoder = AutoModel.from_pretrained(base_model)
        QLoraPipeline(self.lora).apply_lora(encoder, quantize_4bit=quantize_4bit)
        self.model = _Model(encoder, encoder.config.hidden_size, len(self.labels))
        self.history: List[Dict[str, float]] = []

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.model.named_parameters() if "lora_" in n or n.startswith("head.")]

    def _batch(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        contexts: Optional[Sequence[Context]] = None,
        epochs: int = 3,
        lr: float = 2e-3,
        batch_size: int = 16,
        seed: int = 0,
        validation: Optional[tuple] = None,
        patience: int = 2,
    ) -> "LoraClassifier":
        """Train adapters and head. With ``validation=(texts, labels, contexts)`` the accuracy on
        it is measured after every epoch, training stops after ``patience`` epochs without
        improvement and the best epoch's adapters are kept (early stopping: pick the epoch on
        data that is neither training nor final evaluation)."""
        torch.manual_seed(seed)
        rng = random.Random(seed)
        contexts = contexts or [None] * len(texts)
        rows = [with_context(t, c) for t, c in zip(texts, contexts)]
        index = {label: i for i, label in enumerate(self.labels)}
        y = torch.tensor([index[label] for label in labels])
        counts = torch.bincount(y, minlength=len(self.labels)).float()
        weights = len(y) / (len(self.labels) * counts.clamp(min=1))
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(self.trainable_parameters(), lr=lr, weight_decay=0.01)
        order = list(range(len(rows)))
        best_acc, best_state, since_best = -1.0, None, 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            total, n, start = 0.0, 0, time.perf_counter()
            for b in range(0, len(order), batch_size):
                idx = order[b : b + batch_size]
                batch = self._batch([rows[i] for i in idx])
                loss = loss_fn(self.model(**batch), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
                n += len(idx)
            record: Dict[str, float] = {"epoch": epoch, "loss": total / n, "seconds": time.perf_counter() - start}
            if validation is not None:
                v_texts, v_labels, v_contexts = validation
                probs = self.predict_proba(v_texts, v_contexts)
                acc = float(np.mean([self.labels[int(i)] == g for i, g in zip(probs.argmax(1), v_labels)]))
                record["val_accuracy"] = acc
                self.model.train()
                if acc > best_acc:
                    best_acc, since_best = acc, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items() if "lora_" in k or k.startswith("head.")}
                else:
                    since_best += 1
            self.history.append(record)
            if validation is not None and since_best >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], contexts: Optional[Sequence[Context]] = None, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        contexts = contexts or [None] * len(texts)
        rows = [with_context(t, c) for t, c in zip(texts, contexts)]
        out = []
        for b in range(0, len(rows), batch_size):
            logits = self.model(**self._batch(rows[b : b + batch_size]))
            out.append(torch.softmax(logits, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, text: str, context: Context = None) -> "tuple[str, float]":
        p = self.predict_proba([text], [context])[0]
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        state = {n: p.detach().cpu() for n, p in self.model.named_parameters() if "lora_" in n or n.startswith("head.")}
        torch.save(state, os.path.join(directory, "adapters.pt"))
        with open(os.path.join(directory, "model.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "kind": KIND,
                    "labels": self.labels,
                    "base_model": self.base_model,
                    "lora": self.lora.model_dump(),
                    "quantize_4bit": self.quantize_4bit,
                    "max_length": self.max_length,
                    "history": self.history,
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str, **kwargs: Any) -> "LoraClassifier":
        with open(os.path.join(directory, "model.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        clf = cls(
            info["labels"],
            base_model=info["base_model"],
            lora=LoraConfig(**info["lora"]),
            quantize_4bit=info["quantize_4bit"],
            max_length=info["max_length"],
            **kwargs,
        )
        state = torch.load(os.path.join(directory, "adapters.pt"), map_location="cpu")
        missing = clf.model.load_state_dict(state, strict=False)
        unexpected = [k for k in missing.unexpected_keys]
        if unexpected:
            raise ValueError(f"adapter file does not match the model: {unexpected[:3]}")
        clf.history = info.get("history", [])
        clf.model.eval()
        return clf

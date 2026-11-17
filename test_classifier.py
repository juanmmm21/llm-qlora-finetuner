"""LoRA/QLoRA classifier training on a tiny random BERT (downloads ~1 MB from Hugging Face)."""

import tempfile
import unittest

import numpy as np
import torch

from llm_qlora_finetuner.classifier import LoraClassifier, with_context
from llm_qlora_finetuner.finetuner import LoraConfig, LoraLinear

TINY = "hf-internal-testing/tiny-random-bert"


def toy():
    texts = [f"the {c} {w}" for c in ("red", "crimson") for w in ("apple", "car", "sky", "door", "lamp")]
    texts += [f"the {c} {w}" for c in ("blue", "azure") for w in ("apple", "car", "sky", "door", "lamp")]
    labels = ["warm"] * 10 + ["cold"] * 10
    return texts, labels


class TestLoraClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lora = LoraConfig(r=4, alpha=8, target_modules=["query", "value"], dropout=0.0)

    def test_only_adapters_and_head_train_and_base_is_nf4(self) -> None:
        clf = LoraClassifier(["warm", "cold"], base_model=TINY, lora=self.lora, quantize_4bit=True)
        trainable = {n for n, p in clf.model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in n or n.startswith("head.") for n in trainable))
        lora_layers = [m for m in clf.model.modules() if isinstance(m, LoraLinear)]
        self.assertTrue(lora_layers and all(m.quantize_4bit for m in lora_layers))

    def test_learns_and_roundtrips(self) -> None:
        texts, labels = toy()
        clf = LoraClassifier(["warm", "cold"], base_model=TINY, lora=self.lora, quantize_4bit=True)
        clf.fit(texts, labels, epochs=30, lr=5e-3, batch_size=8)
        self.assertLess(clf.history[-1]["loss"], clf.history[0]["loss"])
        acc = np.mean([clf.predict(t)[0] == y for t, y in zip(texts, labels)])
        self.assertGreaterEqual(acc, 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            clf.save(tmp)
            again = LoraClassifier.load(tmp)
            np.testing.assert_allclose(again.predict_proba(texts[:4]), clf.predict_proba(texts[:4]), atol=1e-5)

    def test_context_tags(self) -> None:
        self.assertEqual(with_context("sigue", ["active_project"]), "[active_project] sigue")
        self.assertEqual(with_context("hola", None), "hola")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()

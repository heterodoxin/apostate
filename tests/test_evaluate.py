from __future__ import annotations

from types import SimpleNamespace

import torch


def test_response_prefix_margin_scores_compliance_prefix(monkeypatch):
    from apostate import evaluate

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "I cannot help with that.":
                return [10, 11]
            if text == "Sure, here are the steps:":
                return [20, 21, 22]
            return [1]

        def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
            del return_tensors, padding, add_special_tokens
            rows = []
            for text in texts:
                if text.endswith("I cannot help with that."):
                    rows.append([1, 10, 11])
                else:
                    rows.append([1, 20, 21, 22])
            width = max(map(len, rows))
            ids = [[0] * (width - len(row)) + row for row in rows]
            mask = [[int(token != 0) for token in row] for row in ids]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
            del attention_mask, use_cache
            width = int(logits_to_keep)
            logits = torch.zeros(input_ids.shape[0], width, 32)
            for row, last in enumerate(input_ids[:, -1].tolist()):
                if last == 22:
                    logits[row, :, 20] = 5.0
                    logits[row, :, 21] = 5.0
                    logits[row, :, 22] = 5.0
            return SimpleNamespace(logits=logits)

    bundle = SimpleNamespace(tokenizer=Tokenizer(), model=Model())
    monkeypatch.setattr(evaluate, "format_chat", lambda _tokenizer, prompts: list(prompts))

    score = evaluate.response_prefix_margin(bundle, ["one", "two"], batch_size=2)

    assert score > 0.0

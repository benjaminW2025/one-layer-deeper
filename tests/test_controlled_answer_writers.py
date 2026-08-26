from __future__ import annotations

import unittest

import torch

from benchmark import ModelSpec
from controlled_experiments.run_multiplication import (
    CausalAnswerBlock,
    CausalTransformerAnswerWriter,
    GRUAnswerWriter,
    answer_slot_layout,
    load_submission,
)


class ControlledAnswerWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(74)
        self.source = load_submission()
        self.source.TIE_EMBEDDINGS = False
        self.spec = ModelSpec(17, 14, 500_000_000)
        # Direct-squaring format: X digits ANS followed by OUT=4 slots. The
        # shorter row is padded after its six answer digits.
        self.input_ids = torch.tensor(
            [
                [2, 8, 9, 10, 3, 4, 4, 4, 4, 4, 4, 0, 0, 0],
                [2, 8, 9, 10, 11, 3, 4, 4, 4, 4, 4, 4, 4, 4],
            ]
        )
        self.mask = self.input_ids != 0

    def test_answer_slot_layout_handles_variable_width(self) -> None:
        positions, valid = answer_slot_layout(self.input_ids, 4, 8)
        self.assertEqual(positions[0, :6].tolist(), [5, 6, 7, 8, 9, 10])
        self.assertEqual(valid[0].tolist(), [True] * 6 + [False] * 2)
        self.assertEqual(positions[1].tolist(), list(range(6, 14)))
        self.assertTrue(valid[1].all().item())

    def test_writers_forward_and_backward(self) -> None:
        models = (
            GRUAnswerWriter(self.source, self.spec, 1, 1, 8, 4),
            CausalTransformerAnswerWriter(
                self.source, self.spec, 1, 1, 8, 4, writer_layers=2
            ),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                logits, auxiliary = model(self.input_ids, self.mask)
                self.assertEqual(logits.shape, (2, 14, 17))
                self.assertIsNone(auxiliary)
                loss = logits[self.mask].float().square().mean()
                loss.backward()
                self.assertIsNotNone(model.answer_query.grad)
                self.assertTrue(torch.isfinite(model.answer_query.grad).all())

    def test_causal_block_prevents_higher_digits_affecting_lower_digits(self) -> None:
        block = CausalAnswerBlock(width=16, heads=4).eval()
        answer = torch.randn(2, 8, 16)
        valid = torch.ones(2, 8, dtype=torch.bool)
        context = torch.randn(2, 5, 16)
        context_mask = torch.ones(2, 5, dtype=torch.bool)
        changed = answer.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:])
        with torch.no_grad():
            original_output = block(answer, valid, context, context_mask)
            changed_output = block(changed, valid, context, context_mask)
        torch.testing.assert_close(original_output[:, :4], changed_output[:, :4])


if __name__ == "__main__":
    unittest.main()

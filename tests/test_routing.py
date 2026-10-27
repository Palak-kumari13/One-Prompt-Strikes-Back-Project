"""Run: python -m unittest discover -s tests -v"""
import unittest
import torch
from task1.expert_choice import expert_choice_mask, pack_selected_prompts, utilization_report


class ExpertChoiceTests(unittest.TestCase):
    def test_capacity_and_balance(self):
        scores = torch.randn(16, 3, 1, 8)
        selection, capacity = expert_choice_mask(scores, original_topk=2)
        self.assertEqual(capacity, 4)
        self.assertTrue(torch.all(selection.sum(dim=0) == 4))
        self.assertEqual(int(selection.sum()), 16 * 3 * 2)
        self.assertEqual(utilization_report(selection)['expert_utilization_cv'], 0.0)

    def test_expert_picks_best_inputs(self):
        scores = torch.arange(8.0).view(8, 1, 1, 1).expand(8, 1, 1, 2)
        selection, capacity = expert_choice_mask(scores, 1)
        self.assertEqual(capacity, 4)
        self.assertTrue(selection[4:].all())
        self.assertFalse(selection[:4].any())

    def test_zero_expert_inputs_and_valid_padding(self):
        scores = torch.arange(8.0).view(8, 1, 1, 1).expand(8, 1, 1, 2)
        selection, _ = expert_choice_mask(scores, 1)
        keys = torch.randn(8, 1, 2, 4, requires_grad=True)
        values = torch.randn(8, 1, 2, 4, requires_grad=True)
        packed_k, packed_v, valid = pack_selected_prompts(keys, values, selection)
        self.assertEqual(packed_k.shape, packed_v.shape)
        self.assertFalse(valid[0].any())
        self.assertTrue(valid[-1].any())
        (packed_k.sum() + packed_v.sum()).backward()
        self.assertIsNotNone(keys.grad)

    def test_capacity_capping(self):
        selection, capacity = expert_choice_mask(torch.randn(2, 1, 1, 8), 2)
        self.assertEqual(capacity, 1)  # ceiling to a feasible minimum
        self.assertEqual(int(selection.sum()), 8)


if __name__ == '__main__':
    unittest.main()

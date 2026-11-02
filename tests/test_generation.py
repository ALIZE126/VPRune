"""Exercise VPRune's cached decode using a real tiny Qwen2 (random weights).

These tests check wiring, NOT task accuracy. The separate real-image smoke
test uses a pretrained official checkpoint.
"""
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from vprune.core import VPRuneConfig
from vprune.inference import IMAGE_TOKEN_INDEX, generate_from_features, build_prompt


class GenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          max_position_embeddings=512)
        cfg._attn_implementation = 'eager'
        self.model = Qwen2ForCausalLM(cfg).eval()
        self.ids = torch.tensor([[2, 3, IMAGE_TOKEN_INDEX, 4, 5]])
        self.features = torch.randn(1, 8, 16)

    def test_cached_decode_matches_uncached_original_position_decode(self):
        seen_positions, cached_logits = [], []
        def capture(module, args, kwargs, output):
            seen_positions.append(kwargs['position_ids'].clone())
            cached_logits.append(self.model.lm_head(output.last_hidden_state[:, -1]).detach())
        handle = self.model.model.register_forward_hook(capture, with_kwargs=True)
        try:
            result = generate_from_features(self.model, self.ids, self.features,
                                           VPRuneConfig(budget=.5), max_new_tokens=3, eos_token_id=None)
        finally:
            handle.remove()
        self.assertEqual(len(seen_positions), 3)
        self.assertEqual(seen_positions[1].tolist(), [[12]])
        self.assertEqual(seen_positions[2].tolist(), [[13]])
        self.assertEqual(result.position_ids[0, -2:].tolist(), [10, 11])
        keep = result.pruning.keep_indices
        self.assertEqual(result.position_ids[0, 2:6].tolist(), (keep + 2).tolist())
        embeds = torch.cat([self.model.model.embed_tokens(self.ids[:, :2]), result.pruning.features,
                            self.model.model.embed_tokens(self.ids[:, 3:])], dim=1)
        positions = result.position_ids.clone()
        expected = []
        with torch.no_grad():
            for step in range(3):
                logits = self.model(inputs_embeds=embeds, position_ids=positions, use_cache=False).logits
                torch.testing.assert_close(cached_logits[step], logits[:, -1], rtol=1e-5, atol=1e-7)
                token = logits[:, -1].argmax(-1, keepdim=True)
                expected.append(token.item())
                embeds = torch.cat([embeds, self.model.model.embed_tokens(token)], dim=1)
                positions = torch.cat([positions, torch.tensor([[12 + step]])], dim=1)
        self.assertEqual(result.token_ids.tolist(), expected)
        self.assertEqual(result.next_position, 12)

    def test_eos_stops_without_extra_decode(self):
        first = generate_from_features(self.model, self.ids, self.features, max_new_tokens=1)
        stopped = generate_from_features(self.model, self.ids, self.features, max_new_tokens=8,
                                         eos_token_id=[first.token_ids[0].item()])
        self.assertEqual(stopped.token_ids.numel(), 0)

    def test_requires_exactly_one_image_placeholder(self):
        for ids in [torch.tensor([[2, 3]]), torch.tensor([[-200, -200]])]:
            with self.assertRaises(ValueError):
                generate_from_features(self.model, ids, self.features, max_new_tokens=2)

    def test_invalid_generation_length(self):
        with self.assertRaises(ValueError):
            generate_from_features(self.model, self.ids, self.features, max_new_tokens=0)

    def test_nonfinite_logits_raise_instead_of_generating_token_zero(self):
        # A real corrupted output projection reproduces argmax(NaN) -> token 0.
        with torch.no_grad():
            self.model.lm_head.weight.fill_(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            generate_from_features(self.model, self.ids, self.features, max_new_tokens=2)

    def test_prompt_matches_qwen2_template_without_editing_question(self):
        self.assertEqual(build_prompt('What temperature?'),
                         '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                         '<|im_start|>user\n<image>\nWhat temperature?<|im_end|>\n'
                         '<|im_start|>assistant\n')
        with self.assertRaises(ValueError):
            build_prompt('Look at <image>')


if __name__ == '__main__':
    unittest.main()

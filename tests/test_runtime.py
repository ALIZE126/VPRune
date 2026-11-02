import unittest
from types import SimpleNamespace
from PIL import Image
from transformers import CLIPImageProcessor
from vprune.runtime import prepare_image, resolve_revision
from vprune.inference import tokenize_prompt


class RuntimeTests(unittest.TestCase):
    def test_padding_preserves_whole_image_and_black_border(self):
        processor = CLIPImageProcessor(size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8},
                                       image_mean=[0., 0., 0.], image_std=[1., 1., 1.])
        pixels = prepare_image(Image.new('RGB', (8, 4), (255, 0, 0)), processor,
                               SimpleNamespace(image_aspect_ratio='pad'))
        self.assertEqual(tuple(pixels.shape), (1, 3, 8, 8))
        self.assertEqual(pixels[0, :, :2].abs().sum().item(), 0)
        self.assertEqual(pixels[0, 0, 2:6].sum().item(), 32)

    def test_unsupported_preprocessing_fails_loudly(self):
        with self.assertRaises(ValueError):
            prepare_image(Image.new('RGB', (8, 4)), None, SimpleNamespace(image_aspect_ratio='anyres'))

    def test_official_remote_is_immutable_and_unknown_is_rejected(self):
        self.assertRegex(resolve_revision('apple/FastVLM-1.5B', None), '^[0-9a-f]{40}$')
        with self.assertRaises(ValueError):
            resolve_revision('other/unreviewed-code', None)
        with self.assertRaises(ValueError):
            resolve_revision('apple/FastVLM-1.5B', 'main')

    def test_tokenizer_keeps_one_bos_and_one_image_slot(self):
        class Tokenizer:
            bos_token_id = 1
            def __call__(self, text):
                return SimpleNamespace(input_ids=[1, 2] if text.startswith('<|im_start|>system') else [1, 3, 4])
        self.assertEqual(tokenize_prompt(Tokenizer(), 'test').tolist(), [[1, 2, -200, 3, 4]])


if __name__ == '__main__':
    unittest.main()

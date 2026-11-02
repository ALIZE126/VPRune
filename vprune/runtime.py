"""Official FastVLM loader. Upstream model code/weights are NOT vendored."""
from pathlib import Path
import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

OFFICIAL_REVISIONS = {
    'apple/FastVLM-0.5B': '16375720c2d673fa583e57e9876afde27549c7d0',
    'apple/FastVLM-1.5B': 'dd6608dfa0e17b050e1dde2856c3437fcba197ac',
}


def resolve_revision(model_id, revision):
    if Path(model_id).is_dir():
        if revision:
            raise ValueError('--revision applies to Hub models, not local snapshots')
        return None
    if model_id not in OFFICIAL_REVISIONS:
        raise ValueError('supported Hub models: ' + ', '.join(OFFICIAL_REVISIONS))
    revision = revision or OFFICIAL_REVISIONS[model_id]
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('use an immutable 40-character revision SHA, not a branch/tag')
    return revision


def load_model(model_id, revision=None, device='cpu', dtype=torch.float32,
               trust_remote_code=False, local_files_only=False):
    if not trust_remote_code:
        raise ValueError('FastVLM requires official custom model code. Review it, then pass --trust-remote-code.')
    revision = resolve_revision(model_id, revision)
    options = dict(revision=revision, trust_remote_code=True, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **options)
    model, info = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager',
        output_loading_info=True, **options,
    )
    if any(info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise ValueError('Checkpoint does not match the clean FastVLM architecture: ' + str(info))
    if model.config.model_type != 'llava_qwen2' or not hasattr(model, 'encode_images'):
        raise ValueError('this runner supports the official Qwen2-based FastVLM architecture only')
    if getattr(model.config, 'mm_use_im_start_end', False):
        raise ValueError('image start/end token checkpoints are not supported by this prompt template')
    model.eval().requires_grad_(False)
    tower = model.get_vision_tower()
    if not tower.is_loaded:
        # Loading a new random tower *after* checkpoint load must never be silent.
        raise ValueError('vision tower was not instantiated during checkpoint loading')
    model.to(device=device, dtype=dtype)
    return tokenizer, model, tower.image_processor, revision


def prepare_image(image, processor, config):
    """Same square-padding convention as the original FastVLM Qwen2 path."""
    if getattr(config, 'image_aspect_ratio', None) != 'pad':
        raise ValueError('only image_aspect_ratio=pad is supported; no dynamic-resolution tiling')
    image = image.convert('RGB')
    width, height = image.size
    if width != height:
        size = max(width, height)
        bg = tuple(int(x * 255) for x in processor.image_mean)
        canvas = Image.new('RGB', (size, size), bg)
        canvas.paste(image, ((size - width) // 2, (size - height) // 2))
        image = canvas
    return processor.preprocess(image, return_tensors='pt')['pixel_values']

"""Public CLI: python -m vprune --help."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from PIL import Image
from .core import VPRuneConfig
from .inference import generate, tokenize_prompt
from .runtime import load_model, prepare_image


def main():
    parser = argparse.ArgumentParser(description='VPRune on official FastVLM: VDS + PPR + STR.')
    parser.add_argument('--model', default='apple/FastVLM-1.5B', help='official Hub ID or complete HF snapshot directory')
    parser.add_argument('--revision', help='immutable Hub revision SHA (official defaults are pinned)')
    parser.add_argument('--trust-remote-code', action='store_true', help='explicit consent to execute upstream FastVLM model code')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--question', required=True)
    parser.add_argument('--budget', type=float, default=.3)
    parser.add_argument('--gamma', type=float, default=.3)
    parser.add_argument('--tau', type=float, default=.1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--output', type=Path, help='optional JSON output (contains your question/answer)')
    args = parser.parse_args()
    try:
        config = VPRuneConfig(args.budget, args.gamma, args.tau)
        if args.max_new_tokens < 1:
            raise ValueError('--max-new-tokens must be positive')
        if not args.image.is_file():
            raise ValueError('image file does not exist')
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        dtype_name = args.dtype if args.dtype != 'auto' else ('float16' if device == 'cuda' else 'float32')
        dtype = getattr(torch, dtype_name)
        if device == 'cpu' and dtype != torch.float32:
            raise ValueError('use float32 for CPU inference')
        print('Loading official FastVLM (first run may download weights)...', file=sys.stderr, flush=True)
        tokenizer, model, processor, revision = load_model(
            args.model, args.revision, device, dtype, args.trust_remote_code, args.local_files_only)
        with Image.open(args.image) as source:
            image = source.convert('RGB')
        pixels = prepare_image(image, processor, model.config).to(device=device, dtype=dtype)
        ids = tokenize_prompt(tokenizer, args.question).to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        result = generate(model, ids, pixels, config, args.max_new_tokens, tokenizer.eos_token_id)
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        answer = tokenizer.decode(result.token_ids, skip_special_tokens=True).strip()
        import transformers
        report = {
            'method': 'VPRune', 'modules': ['VDS', 'PPR', 'STR'],
            'model': Path(args.model).name if Path(args.model).is_dir() else args.model,
            'revision': revision or getattr(model.config, '_commit_hash', None),
            'question': args.question, 'answer': answer,
            'image_name': args.image.name, 'image_pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
            'image_size': list(image.size), 'pixel_tensor_shape': list(pixels.shape),
            'budget': config.budget, 'gamma': config.gamma, 'tau': config.tau,
            'preserve_positions': True, 'text_relevance': False,
            'visual_tokens_before': result.pruning.original_tokens,
            'visual_tokens_after': result.pruning.keep_indices.numel(),
            'updated_tokens': result.pruning.updated_tokens,
            'keep_indices': result.pruning.keep_indices.cpu().tolist(),
            'position_ids': result.position_ids[0].cpu().tolist(),
            'next_decode_position': result.next_position,
            'generated_ids': result.token_ids.cpu().tolist(), 'max_new_tokens': args.max_new_tokens,
            'device': device, 'dtype': dtype_name, 'pruning_device': result.pruning.pruning_device,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'elapsed_seconds_excluding_load': round(time.perf_counter() - start, 4),
            'timing_note': 'Single smoke run, not a latency benchmark.',
        }
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'VPRune error: {exc}\n')


if __name__ == '__main__':
    main()

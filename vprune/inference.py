"""Single-image, batch-1 greedy generation; explicit PPR during cached decode."""
from dataclasses import dataclass
import torch
from .core import VPRuneConfig, PruningResult, prune_visual_tokens, original_position_ids

IMAGE_TOKEN_INDEX = -200


def _greedy_token(model, hidden):
    logits = model.lm_head(hidden[:, -1])
    if not torch.isfinite(logits).all():
        raise RuntimeError('non-finite LLM logits; use float32 on CPU/MPS and check model numerics')
    return logits.argmax(-1, keepdim=True)


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    pruning: PruningResult
    position_ids: torch.Tensor
    next_position: int


def build_prompt(question):
    if not isinstance(question, str) or not question.strip() or '<image>' in question:
        raise ValueError("provide a nonempty question without an <image> placeholder")
    return ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\n<image>\n' + question + '<|im_end|>\n'
            '<|im_start|>assistant\n')


def tokenize_prompt(tokenizer, question):
    """Match the Qwen2/LLaVA split-at-image convention, including optional BOS."""
    before, after = build_prompt(question).split('<image>')
    left, right = tokenizer(before).input_ids, tokenizer(after).input_ids
    offset = int(bool(left) and left[0] == tokenizer.bos_token_id)
    return torch.tensor([left + [IMAGE_TOKEN_INDEX] + right[offset:]], dtype=torch.long)


@torch.inference_mode()
def generate_from_features(model, input_ids, visual_features, config=None,
                           max_new_tokens=16, eos_token_id=None):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a (1,L) torch.long tensor")
    slots = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
    if slots.numel() != 1:
        raise ValueError("exactly one image placeholder is required")
    if input_ids.device != visual_features.device:
        raise ValueError("input_ids and visual_features must share a device")
    config = config or VPRuneConfig()
    p = int(slots.item())
    pruning = prune_visual_tokens(visual_features, config)
    qwen = model.model
    before = qwen.embed_tokens(input_ids[:, :p])
    after = qwen.embed_tokens(input_ids[:, p + 1:])
    embeds = torch.cat([before, pruning.features, after], dim=1)
    positions = original_position_ids(p, pruning.keep_indices, pruning.original_tokens, after.shape[1])
    next_pos = p + pruning.original_tokens + after.shape[1]
    context_limit = getattr(model.config, 'max_position_embeddings', None)
    if context_limit and next_pos + max_new_tokens > context_limit:
        raise ValueError("original position span plus generation exceeds model context limit")
    out = qwen(inputs_embeds=embeds, position_ids=positions, use_cache=True, return_dict=True)
    cache = out.past_key_values
    nxt = _greedy_token(model, out.last_hidden_state)
    stop_ids = set() if eos_token_id is None else set(eos_token_id if isinstance(eos_token_id, (tuple, list)) else [eos_token_id])
    generated = []
    for step in range(max_new_tokens):
        token = int(nxt.item())
        if token in stop_ids:
            break
        generated.append(token)
        if step + 1 == max_new_tokens:
            break
        pos = torch.tensor([[next_pos + step]], dtype=torch.long, device=input_ids.device)
        out = qwen(inputs_embeds=qwen.embed_tokens(nxt), position_ids=pos,
                   past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        nxt = _greedy_token(model, out.last_hidden_state)
    return GenerationResult(torch.tensor(generated, dtype=torch.long, device=input_ids.device),
                            pruning, positions, next_pos)


@torch.inference_mode()
def generate(model, input_ids, pixel_values, config=None, max_new_tokens=16, eos_token_id=None):
    """Vision encoder -> projector -> VDS/STR -> PPR -> LLM. No monkeypatches."""
    features = model.encode_images(pixel_values)
    return generate_from_features(model, input_ids, features, config, max_new_tokens, eos_token_id)

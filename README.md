# VPRune

**Training-free visual token pruning for FastVLM: visual-only diversity selection, position preservation, and similarity-guided token recycling.**

这是仅包含 VPRune 最终方法的独立研究代码。无需安装原实验仓库，也不包含其他裁剪方法、训练控制器、私有数据、服务器配置或模型权重。

## Method

`Image → FastViT-HD → Projector → VDS → STR → PPR position IDs → Qwen2 LLM`

- **VDS — Visual-only Diversity Selection**：对 projector 输出的视觉特征归一化，计算 cosine Gram kernel，使用 greedy DPP MAP 选择互补特征。没有 text relevance，也没有视觉 attention/saliency 加权。
- **PPR — Position-Preserving Restoration**：保留 token 按原始顺序排列，并保留原始位置编号。图像后文本和生成位置沿用未裁剪序列的位置跨度。
- **STR — Similarity-guided Token Recycling**：每个被删 token 分配到 cosine 最相似的保留 token，按接收组做 softmax，然后融合特征。无来源分配的保留 token 不变。这里的相似性不是二维空间邻近。

默认 `budget=0.3, gamma=0.3, tau=0.1`。支持改变 budget；最终方法始终保留 VDS/PPR。

```text
K = max(1, min(N, round(budget * N)))
L[i,j] = cosine(x[i], x[j])
S ≈ argmax_{|S|=K} log det(L[S])       # greedy incremental Cholesky
owner(i) = argmax_{j∈S} L[i,j]
w[i,j] = softmax_{i∈G[j]}(L[i,j] / tau)
fused[j] = (1-gamma)*x[j] + gamma*sum_{i∈G[j]} w[i,j]*x[i]
visual_position[j] = N_prefix + original_index[j]
first_suffix_position = N_prefix + N
```

`round` 为 Python 的 round，而非 floor；当 N=256 时，b=0.1/0.3/0.6/0.8 对应 K=26/77/154/205。PPR 不插入空白 token，实际输入仍只有 K 个视觉向量。STR 不重建图像像素，也不改变位置编号。

## Install

推荐 Python 3.10 或3.11。依赖固定在 PyTorch 2.6.0 / Transformers 4.48.3 / timm 1.0.15，避免模型自定义代码随依赖升级改变行为。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

CUDA 请安装与你的驱动匹配的 PyTorch 2.6.0 / torchvision 0.21.0 构建。CPU、Apple MPS 默认 float32，CUDA 默认 float16。本机 MPS float16 在 Qwen2 首层出现 NaN，float32 已验证正常；生成器会明确拒绝非有限 logits。MPS 不支持 float64，因此仅小规模 VDS/STR 阶段移至 CPU，模型仍在 MPS；不要用这个兼容路径代表论文板端时延。

## Run one image

```bash
python examples/create_demo.py
python -m vprune \
  --model apple/FastVLM-1.5B \
  --trust-remote-code \
  --image examples/demo.png \
  --question "What color is the circle? Answer with one word." \
  --budget 0.3 --gamma 0.3 --tau 0.1 \
  --max-new-tokens 16 \
  --output runs/demo.json
```

也可使用 `vprune` 命令替代 `python -m vprune`，或把 `--image` 换成自己的图片。问题字符串原样保留，不自动修改大小写或添加评测提示词。示例图由本项目脚本绘制，不使用第三方数据集图像。

该命令会加载官方预训练权重，执行完整视觉编码、投影、VPRune、LLM prefill 和 greedy decode。输出包括答案、256→K 的 token 数量、keep indices、position IDs、实际发生融合的 token 数量及版本信息。单次用时仅作为通路记录，不是时延基准。

本地验证：2026-11-02，独立环境，官方1.5B固定版本，示例图含红色方形与蓝色圆形，上述英文问题的实际输出为 `blue`。CPU float32 与 MPS float32 均通过；256→77 tokens，VDS/PPR固定、γ=.3、τ=.1。完整回归测试使用 `python -m unittest discover -s tests -v`。这是一张原创简单图的通路测试，不是TextVQA/GQA准确率结果。

### Checkpoints and upstream code

支持两个官方 Hugging Face checkpoint，默认固定为以下不可变版本：

- `apple/FastVLM-1.5B`：`dd6608dfa0e17b050e1dde2856c3437fcba197ac`
- `apple/FastVLM-0.5B`：`16375720c2d673fa583e57e9876afde27549c7d0`

首次运行会下载相应权重到 Hugging Face 缓存。`--trust-remote-code` 表示明确授权执行官方 FastVLM 自定义模型代码；请先阅读对应版本的代码。没有该选项会明确拒绝加载。不会默认信任任意第三方 Hub 仓库或移动的 `main` 分支。

离线使用：先缓存官方模型，再加 `--local-files-only`，或用 `--model /path/to/complete/hf/snapshot`。本地目录需要完整权重、tokenizer、config 和 `llava_qwen.py`；旧实验目录中仅有 config/tokenizer 不是可用 checkpoint。此发行版不直接加载带实验控制器或 LoRA 的 checkpoint，不会静默忽略不匹配权重。

官方 HF checkpoint 与历史实验使用的自定义 checkpoint **不能未经核验当成相同权重**。本项目的本地 smoke test 是通路验证，不宣称复现全部论文精度。CPU/MPS/CUDA、精度和数值并列情况可能影响 DPP 选择或生成。

## Python API

```python
from vprune import VPRuneConfig, prune_visual_tokens
from vprune.inference import generate

config = VPRuneConfig(budget=0.3, gamma=0.3, tau=0.1)
# Post-projector features: [1, N, D]
result = prune_visual_tokens(visual_features, config)
# An already loaded official FastVLM model, prompt with one image placeholder:
generated = generate(model, input_ids, pixel_values, config,
                     max_new_tokens=16, eos_token_id=tokenizer.eos_token_id)
```

支持单张图片、batch size 1、官方 pad 预处理、greedy decoding。不包含多图、video、beam search、dynamic-resolution tiling、训练和全量评测框架。

`gamma=0` 可禁用 STR 作对照，VDS/PPR 不变；`budget=1` 为完整视觉输入，STR 无被删来源，特征保持不变。`tau` 必须至少为 `1e-6`：原实验会静默截断更小的值，此发行版明确拒绝它们，避免报告值与实际计算不符。混合精度 GPU scatter 累加存在舍入差异，不保证跨设备 bitwise 一致。

## Files

```text
vprune/core.py        VDS / STR / original position IDs
vprune/inference.py   Qwen2 prompt and explicit cached greedy decode
vprune/runtime.py     pinned official checkpoint loader and image preprocessing
vprune/cli.py         command-line entry point and JSON trace
tests/               numerical, sequence and runtime contract tests
examples/            reproducible original demo image generator
scripts/             source-only release packaging
```

## Attribution and licenses

本项目源代码使用 [Apache-2.0](LICENSE)。它不改变任何上游组件或模型权重的许可证。FastVLM 官方代码及模型权重分别遵守其上游条款；Apple 模型权重的研究用途限制仍适用。见 [NOTICE](NOTICE) 及 [官方模型许可](https://huggingface.co/apple/FastVLM-1.5B/blob/dd6608dfa0e17b050e1dde2856c3437fcba197ac/LICENSE)。

保留以下来源署名，不将这些已有组件宣称为全新发明：

- [FastVLM](https://github.com/apple/ml-fastvlm)：基础模型及视觉编码器。
- [CDPruner](https://arxiv.org/abs/2506.10967)：DPP 条件多样性选择框架；本实现去掉 relevance 条件项。
- [Chen et al., 2018](https://arxiv.org/abs/1709.05135)：快速 greedy DPP MAP 算法。
- [GAP](https://arxiv.org/abs/2506.21873)：保留原始 position IDs。

论文标题、作者和正式引用尚未填写；不生成虚构 BibTeX。此目录不包含历史 accuracy 表、板端结果、数据集素材或 SSH 信息。

## Prepare a source archive

```bash
python scripts/package_release.py
```

生成 `dist/vprune-source.zip`，只打包明确列出的代码、测试、说明与原创示例；不包含 `.venv`、模型、日志或 Git 历史。发布前仍应自行确认作者、论文信息及许可证选择。

# Running and evaluating YANchor-4B

[Model overview and results](README.md)

Download the complete model bundle from [Hugging Face](https://huggingface.co/rockmatrix-ai/YANchor-4B) or [ModelScope](https://modelscope.ai/models/RocoreMatrix/YANchor-4B). It contains the weights, tokenizer and evaluation inputs. This GitHub repository contains the inference and evaluation code. Run all commands from the bundle root; `--model` defaults to that directory. The bundle supports standard Transformers loading, the optimized H100 runtime, and native serving-framework adapters.

```text
YANchor-4B/
├── README.md, USAGE.md, CITATION.cff
├── assets/         # Figures used by README.md
├── LICENSE, LICENSE-CODE, NOTICE
├── run.py, requirements.txt, Dockerfile, pyproject.toml
├── configuration_yanchor.py, modeling_yanchor.py
├── vllm_yanchor.py, sglang_models/
├── wheels/         # Pinned public Transformers dependency
├── config.json, generation_config.json, tokenizer*.json
├── model.safetensors.index.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── runtime/        # Model, GPU operators, generation and scoring
├── vision/         # Visual frontend weights and processor
└── evaluation_data/           # MANIFEST.json, text, vision, memory and code tests
```

Use `AutoModelForCausalLM` with `trust_remote_code=True` or `run.py`. Both load the YANchor architecture and its independent memory modules. The unmodified Qwen model class is not a substitute.

## Runtime choices

| Route | Purpose | Verified hardware |
|---|---|---|
| Transformers / `--backend torch` | Standard Python generation without custom CUDA extensions | CPU FP32; H100 BF16 |
| `--backend cuda --profile fast` | Highest-throughput H100 deployment and benchmark reproduction | H100 80 GB |
| Native vLLM adapter | vLLM scheduling and OpenAI-compatible serving | One H100, TP1 |
| Native SGLang adapter | SGLang scheduling and OpenAI-compatible serving | One H100, TP1 |

The PyTorch backend implements the same sliding window, fixed-capacity memory, commit boundaries and recurrent state. It does not require FlashAttention, FLA, TileLang or Triton. The accelerated backend retains the published B1 and B32–B512 profiles. Different kernels and arithmetic orders can change individual outputs. CPU and CUDA are verified; Apple MPS, ROCm and other accelerators have not been validated with the released weights.

## Portable installation and Transformers

Install PyTorch for your platform, then use the bundled Transformers version in a separate environment:

```bash
python -m pip install ./wheels/transformers-5.7.0.dev0-py3-none-any.whl -e .
python -I run.py demo --backend torch --device cpu --prompt 'Compute 17 × 23.'
```

For CUDA without custom extensions, use `--backend torch --device cuda`. `--backend auto` selects the accelerated path on Hopper when its dependencies are available, and otherwise selects PyTorch. FP32 is the CPU default; supported CUDA devices default to BF16.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "."
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=dtype, backend="torch"
).to(device).eval()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Compute 17 × 23."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The standard backend uses bounded prefill chunks, 1,024 tokens by default. Set `prefill_chunk_size` when loading to change that workspace tradeoff. Keep the supplied Transformers version for this route; native serving frameworks use their own Transformers dependencies in separate environments.

## Optimized H100 environment

Use an NVIDIA H100 80 GB, Linux, and the supplied NVIDIA PyTorch container. The base image supplies Python, PyTorch, CUDA and FlashAttention; `requirements.txt` pins the remaining runtime dependencies. The required public Transformers revision is included as a wheel, so installation does not clone GitHub. CUDA extensions build against the container's existing PyTorch. Build on a host with internet access, then transfer the image to the GPU host if needed.

```bash
docker build -t yanchor-runtime .
docker run --rm --gpus all --ipc=host -v "$PWD:/model" yanchor-runtime demo
```

If PyPI downloads are slow, select a mirror with `docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t yanchor-runtime .`. The build context excludes weights and evaluation data.

Inside that image, the equivalent command is `python -I run.py demo`. For the examples below, replace `python -I run.py` with the Docker invocation when running from the host. The B1 service loads the model and prewarms its short-input decode path before reporting READY. New input shapes can still trigger compilation on first use. Standalone commands start a fresh process and include cold-start overhead. Output directories must be writable. For Docker HTTP access on Linux, add `--network host` to the Docker invocation.

## Demo and batch inference

```bash
python -I run.py demo --prompt 'Compute 17 × 23.'
python -I run.py serve --batch 1 --port 8000
```

The service exposes `GET /v1/models` and `POST /v1/chat/completions`, including SSE streaming and token usage. It accepts plain-text messages with one response per request. The lightweight server serializes generation calls; use a native framework for concurrent independent API requests. B1 checks completion every 16 decode steps. The original batch endpoint at `/` accepts a `messages` array or a `prompts` array:

```bash
curl http://127.0.0.1:8000/ -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Compute 17 × 23."}],"max_new_tokens":4096}'
```

A standard streaming request works with either backend:

```bash
python -I run.py serve --backend torch --device cuda --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YANchor-4B","messages":[{"role":"user","content":"Compute 17 × 23."}],"temperature":0,"max_tokens":512,"stream":true,"stream_options":{"include_usage":true}}'
```

All supported deployment sizes use the same entrypoint:

| Batch | Launch option | MLP compute | GDN state |
|---:|---|---|---|
| 1 | `--batch 1 --profile fast` | INT8 GEMV | FP32 |
| 32 | `--batch 32 --profile fast` | BF16 | FP16 |
| 64 | `--batch 64 --profile fast` | BF16 | FP16 |
| 128 | `--batch 128 --profile fast` | BF16 | FP16 |
| 256 | `--batch 256 --profile fast` | BF16 | FP16 |
| 320 | `--batch 320 --profile fast` | BF16 | FP16 |
| 512 | `--batch 512 --profile fast` | BF16 | FP16 |

`--batch` is the per-GPU slot capacity. Short input lists use only the required slots. Supply a list of prompts to the service, or a JSONL file containing one `{"prompt":"..."}` object per line:

```bash
python -I run.py generate --batch 320 --input prompts.jsonl --output outputs/generated
```

The `fast` profile uses CUDA Graphs, packed projections, fused normalization and gates, bounded prefill, fixed-capacity GQA memory and physical sliding-window caches. B1 additionally uses the INT8 MLP GEMV path; B32–B512 use BF16 MLPs and FP16 GDN state. Model weights on disk remain BF16. `--profile reference` disables the B1 INT8 MLP and keeps FP32 GDN state and reproducible sliding-window arithmetic. These arithmetic profiles can yield different individual sampled answers.

Prefill, pure decode and model loading are reported separately. To measure all seven fixed-work batch sizes, invoke the same benchmark command with each batch value. Independent commands may run on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -I run.py benchmark --batch 1 --prefill 4096 --max-new-tokens 4096 --output outputs/b1.json
CUDA_VISIBLE_DEVICES=1 python -I run.py benchmark --batch 320 --prefill 4096 --max-new-tokens 4096 --output outputs/b320.json
```

The benchmark warms up first, disables EOS and mechanical-loop stopping for fixed work, and measures pure decode independently of prefill. Use identical input/output lengths for comparisons: graph setup and completion polling remain in the decode denominator and have a larger effect on short outputs. Cold-start demo timings do not measure warmed sustained decode throughput. B1 uses K1; larger batches use K8 prompt sharing by default. `--k 1` selects unique prompts. Benchmark token sequences are synthetic; they are not capability questions.

## Native serving frameworks

Use a separate environment for each framework. The native adapters retain framework scheduling, attention and GDN execution, with YANchor memory stored in the framework's request slots. Decode accesses memory in place through GPU slot indices and uses CUDA Graphs, packed memory projections and fused normalization. These routes cover text generation on one GPU, TP1, with prefix caching and speculative decoding disabled. Prefill runs eagerly. Vision evaluation continues to use the bundled evaluator.

The commands below enable `YANCHOR_NATIVE_FAST=1`: B1 decode uses the same group-256 INT8 MLP GEMV arithmetic as the specialized runtime; prefill and larger batches keep BF16 MLPs. GDN state remains FP32. vLLM also combines its B1 GDN input projections. Omit this environment variable for BF16 MLP execution. Weights on disk and tokenizer are unchanged. Arithmetic profiles and framework sampling implementations can produce different individual answers.

On one H100 80 GB, a warmed B1 workload with 4,096 input tokens and 2,048 output tokens gave the following results. All routes used temperature 1.0, top-k 20, top-p 0.95, presence penalty 1.5, the fast B1 MLP profile, and ignored EOS for this fixed-workload measurement. Decode excludes prefill, initialization and graph capture.

| Route | Pure decode, token/s | Warm HTTP TTFT, seconds |
|---|---:|---:|
| Specialized CUDA runtime | 212.5 | — |
| SGLang, explicit seed | 205.7 | 0.102 |
| vLLM, explicit seed | 186.9 | 0.126 |
| vLLM, no explicit seed | 193.1 | 0.118 |

The specialized runtime was measured in process; its prefill time was 0.122 seconds. Framework measurements used streamed HTTP responses and timed TTFT to the first text content. Seeded runs used seed 20264924; each framework retains its own sampling implementation. The unseeded vLLM row uses its faster default sampling path. These fixed-workload measurements do not predict every prompt's latency or establish answer equivalence across runtimes.

### vLLM

Verified with vLLM `0.28.1rc1.dev13+g76c0c6530`, PyTorch `2.13.0+cu130`, and Transformers `5.16.1`. In that vLLM environment, install only the adapter; retain the framework's own dependencies:

```bash
python -m pip install --no-deps --no-build-isolation -e .
YANCHOR_NATIVE_FAST=1 VLLM_PLUGINS=yanchor vllm serve "$PWD" \
  --trust-remote-code --served-model-name YANchor-4B \
  --dtype bfloat16 --max-model-len 16384 \
  --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.6 --no-enable-prefix-caching \
  --mamba-cache-mode none --mamba-ssm-cache-dtype float32 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --additional-config '{"gdn_prefill_backend":"triton"}'
```

The plugin is registered through vLLM's model registry. Independent memory is stored in its allocated state-cache pages and follows the framework's slot ownership. The fused normalization retains vLLM's FP32 residual-add arithmetic. Keep prefix caching and speculative decoding disabled for this adapter. This vLLM version uses a slower sampling implementation when a request specifies a seed; the adapter preserves that seeded behavior.

### SGLang

Verified with SGLang `0.5.20`, PyTorch `2.13.0+cu130`, and Transformers `5.12.1`. In that SGLang environment:

```bash
python -m pip install --no-deps --no-build-isolation -e .
YANCHOR_NATIVE_FAST=1 SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_models python -m sglang.launch_server \
  --model-path "$PWD" --trust-remote-code --served-model-name YANchor-4B \
  --dtype bfloat16 --context-length 16384 --chunked-prefill-size 4096 \
  --max-running-requests 4 --max-mamba-cache-size 4 \
  --max-total-tokens 65536 --mem-fraction-static 0.6 \
  --disable-radix-cache --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode 1 2 4 \
  --attention-backend fa3 --page-size 1 \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --mamba-ssm-dtype float32
```

SGLang uses its pure sliding-window cache, recurrent-state pool and overlapping scheduler. Independent memory is registered with the slot lifecycle, including reset and copy operations. The H100 configuration uses FA3 because this SGLang version's pure sliding-window allocator requires page size 1, which is incompatible with its H100 FA4 paged kernel. Set both request and Mamba-slot capacities together; the independent memory allocation grows with the Mamba-slot capacity. The supplied four-request configuration leaves memory headroom for this state.

The native framework commands were exercised with a 15,312-token published memory input, complete EOS-terminated answers, concurrent requests and subsequent slot reuse. This establishes those paths; longer native-framework contexts, tensor/pipeline parallelism, prefix-cache reuse, quantized model loading and speculative decoding are outside the verified configuration. Standard Transformers and the specialized runtime retain their configured context budgets.

## Evaluation

By default, evaluation processes every question supplied for each source, with no additional per-source question limit. Run the complete designated evaluation splits with:

```bash
python -I run.py evaluate --suite text --gpus 0,1,2,3,4,5,6,7 --batch 320 --output outputs/text
python -I run.py evaluate --suite vision --gpus 0,1,2,3,4,5,6,7 --output outputs/vision
python -I run.py evaluate --suite memory --gpus 0,1,2,3,4,5,6,7 --output outputs/memory
python -I run.py status --output outputs/text
```

Use a fresh output directory for each evaluation. The CPU broker distributes attempts among GPU replicas; completed slots are refilled while other answers continue. Workers write response fragments, exit, and then CPU scoring starts. `--generate-only` stops after generation. Existing fragments can be scored with `python -I run.py score --output outputs/text`, without GPUs.

`evaluation_data/MANIFEST.json` lists the designated evaluation splits and their complete question counts. The text evaluation contains **113,743 questions across 29 sources** and expands to **715,267 responses** under automatic K. The report compares all 29 text sources. Auto-K uses the complete source size: fewer than 100 questions → K64; 100–10,000 → K16; above 10,000 → K1. `--sources AIME2024 DROP` selects sources.

Text generation uses temperature 0.6, top-k 20, top-p 0.95, no presence penalty, seed 20260923 and a 131,072-token output cap. The frozen `prompt_text` is the model input; references and executable tests are consumed only by CPU scorers. Every sampled trajectory contributes to the accuracy denominator. Fixed seeds and large K support statistical reproduction; arithmetic order and batch shapes can change individual trajectories.

The visual panel contains 26,668 inputs from seven sources. The report presents six of them; OCRBench is included as an additional evaluation. It uses K1, reference arithmetic, B128, temperature 1, top-k 20, top-p 0.95, presence penalty 1.5, seed 20260912 and a 32,768-token cap. One damaged AI2D item is excluded by the supplied reference correction, leaving 26,667 scored responses. Image processing uses the included visual frontend and a 65,536–4,194,304 pixel range.

Visual media are stored losslessly in `evaluation_data/vision/media.zip` and read directly by the evaluator; extraction is not required.

The memory panel contains only the four tasks in Section 9 of the technical report: exact retrieval, state-update interpretation, cross-document combination and variable binding. Each task has 64 contexts, for 256 contexts and 1,024 queries in total. Contexts form original/counterfactual pairs and span approximately 16K–130K tokens. Evaluation uses greedy K1, reference arithmetic, B1, seed 2026091707 and a 1,024-token output cap. Results include query accuracy, all-four-correct accuracy, the four task breakdowns, and both-correct accuracy for paired queries whose answers should change or remain stable.

### Scoring

All 29 text sources use the v43 answer-verification implementation in `runtime/evaluation/` and `runtime/scoring/`. Final-answer extraction is independent of the reference answer; it recognizes explicit terminal choices and equivalent numeric forms while retaining ambiguity checks. Each benchmark keeps its scoring metric and full questions × K denominator. `RESULTS.json` records `cpu_verifier_revision`. IFEval and IFBench fix the language-detector seed to 0.

| Sources | Rule |
|---|---|
| AIME, HMMT, GSM8K, MATH500 | Reference-independent answer extraction; numeric, structured and symbolic equivalence, with math-verify where needed. |
| ARC, C-Eval, CMMLU, GPQA, HellaSwag, MMLU family, SuperGPQA, WinoGrande | Typed option extraction and comparison against the supplied option layout. Competing commitments do not receive credit. |
| BBH, BBEH | Task-specific deterministic comparison; the official BBEH comparison is included. |
| HumanEval, MBPP | Execute the submitted Python program against the supplied tests. |
| LiveCodeBenchV6 | Official execution backend with the supplied public and hidden tests. |
| IFEval, IFBench | Official strict instruction checks. Whole-prompt success requires every instruction; per-instruction outcomes are retained. |
| DROP | Official span normalization, numeric matching and optimal alignment. F1 is primary; exact match is also reported. |
| MuSR | Option accuracy, macro-averaged over the three task subsets. |
| LogiQA2 | Option accuracy over the supplied English panel. |

The primary metric is `official_primary_mean_pass_at_1` for correctness-based text tasks and `score` for DROP, MuSR and LogiQA2. Source-specific macro averages and the corrected MMLU-Redux view are preserved. Mean accuracy averages the K sampled answers, with all question × K attempts in the denominator; observed pass@K is reported separately.

The final-answer region is preferred. A clearly committed terminal answer in reasoning can retain semantic correctness while failing the strict output contract. Extraction never chooses a candidate because it matches the reference. Unresolved answers receive no success credit and retain diagnostic verdicts. Visual metrics include source accuracy, MMBench circular accuracy, MME category totals and POPE precision/recall/F1, without an external model judge.

Text responses stop at EOS, the token budget or an exact mechanical loop with a period of 128–256 tokens repeated eight times. Bounded suffix checks share completion-check intervals. Visual and memory runs retain their respective stopping rules.

### Outputs and monitoring

`STATUS.json` records expected, claimed and completed attempts. During generation, `total_decode_tokens_per_second` sums recent pure-decode rates across active ranks. `rank-*-metrics.json` separates initial prefill, refill prefill, pure decode, useful output tokens and peak GPU memory. Pure-decode rates exclude prefill and the first token of each response.

`GENERATION_COMPLETE.json` records persisted generation counts. `RESULTS.json` contains scores, denominators and sampling settings after scoring. A partial run is marked `INCOMPLETE`.

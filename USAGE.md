# Running and evaluating YANchor-4B

[Model overview and results](README.md)

This GitHub repository contains the runtime, evaluation code, dependency definitions and required metric resources. Obtain the complete YANchor-4B Hugging Face bundle for model weights, tokenizer, visual frontend and evaluation data. Keep that bundle in a separate directory and pass its path with `--model`.

```text
YANchor/
├── README.md, USAGE.md, CITATION.cff
├── assets/         # Figures used by README.md
├── LICENSE, MODEL_LICENSE, NOTICE
├── run.py, requirements.txt, Dockerfile
├── wheels/         # Pinned public Transformers dependency
└── runtime/
```

The examples below use `/weights` as the model-bundle path. For native Python execution, replace it with the absolute path to your downloaded bundle. For Docker, the command mounts your bundle at `/weights`.

Use `run.py` to load the custom architecture. Loading these weights with the stock Qwen `AutoModelForCausalLM` class does not install the YANchor memory modules.

## Environment

Use an NVIDIA H100 80 GB, Linux, and the supplied NVIDIA PyTorch container. The base image supplies Python, PyTorch, CUDA and FlashAttention; `requirements.txt` pins the remaining runtime dependencies. The required public Transformers revision is included as a wheel, so installation does not clone GitHub. CUDA extensions build against the container's existing PyTorch. Build on a host with internet access, then transfer the image to the GPU host if needed.

```bash
docker build -t yanchor-runtime .
docker run --rm --gpus all --ipc=host \
  -v "$PWD:/model" -v /absolute/path/YANchor-4B:/weights:ro \
  yanchor-runtime demo --model /weights
```

If PyPI downloads are slow, select a mirror with `docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t yanchor-runtime .`. The build context excludes weights and evaluation data.

Inside that image, the equivalent command is `python -I run.py demo --model /weights`. For the examples below, replace `python -I run.py` with the Docker invocation when running from the host. The B1 service loads the model and prewarms its short-input decode path before reporting READY. New input shapes can still trigger compilation on first use. Standalone commands start a fresh process and include cold-start overhead. Output directories must be writable. For Docker HTTP access on Linux, add `--network host` to the Docker invocation.

## Demo and batch inference

```bash
python -I run.py demo --model /weights --prompt 'Compute 17 × 23.'
python -I run.py serve --model /weights --batch 1 --port 8000
```

The HTTP service accepts POST requests with a `messages` array or a `prompts` array and returns the completed response. B1 demo and service requests check completion every 16 decode steps to limit computation after EOS:

```bash
curl http://127.0.0.1:8000/ -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Compute 17 × 23."}],"max_new_tokens":4096}'
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
python -I run.py generate --model /weights --batch 320 --input prompts.jsonl --output outputs/generated
```

The `fast` profile uses CUDA Graphs, packed projections, fused normalization and gates, bounded prefill, fixed-capacity GQA memory and physical sliding-window caches. B1 additionally uses the INT8 MLP GEMV path; B32–B512 use BF16 MLPs and FP16 GDN state. Model weights on disk remain BF16. `--profile reference` disables the B1 INT8 MLP and keeps FP32 GDN state and reproducible sliding-window arithmetic. These arithmetic profiles can yield different individual sampled answers.

Prefill, pure decode and model loading are reported separately. To measure all seven fixed-work batch sizes, invoke the same benchmark command with each batch value. Independent commands may run on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -I run.py benchmark --model /weights --batch 1 --prefill 4096 --max-new-tokens 4096 --output outputs/b1.json
CUDA_VISIBLE_DEVICES=1 python -I run.py benchmark --model /weights --batch 320 --prefill 4096 --max-new-tokens 4096 --output outputs/b320.json
```

The benchmark warms up first, disables EOS and mechanical-loop stopping for fixed work, and measures pure decode independently of prefill. Use identical input/output lengths for comparisons: graph setup and completion polling remain in the decode denominator and have a larger effect on short outputs. Cold-start demo timings do not measure warmed sustained decode throughput. B1 uses K1; larger batches use K8 prompt sharing by default. `--k 1` selects unique prompts. Benchmark token sequences are synthetic; they are not capability questions.

## Evaluation

By default, evaluation processes every question supplied for each source, with no additional per-source question limit. Run the complete designated evaluation splits with:

```bash
python -I run.py evaluate --model /weights --suite text --gpus 0,1,2,3,4,5,6,7 --batch 320 --output outputs/text
python -I run.py evaluate --model /weights --suite vision --gpus 0,1,2,3,4,5,6,7 --output outputs/vision
python -I run.py evaluate --model /weights --suite memory --gpus 0,1,2,3,4,5,6,7 --output outputs/memory
python -I run.py status --output outputs/text
```

Use a fresh output directory for each evaluation. The CPU broker distributes attempts among GPU replicas; completed slots are refilled while other answers continue. Workers write response fragments, exit, and then CPU scoring starts. `--generate-only` stops after generation. Existing fragments can be scored with `python -I run.py score --model /weights --output outputs/text`, without GPUs.

`/weights/evaluation_data/MANIFEST.json` lists the designated evaluation splits and their complete question counts. The text evaluation contains **113,743 questions across 29 sources** and expands to **715,267 responses** under automatic K. The report compares 27 common text sources. Auto-K uses the complete source size: fewer than 100 questions → K64; 100–10,000 → K16; above 10,000 → K1. `--sources AIME2024 DROP` selects sources.

Text generation uses temperature 0.6, top-k 20, top-p 0.95, no presence penalty, seed 20260923 and a 131,072-token output cap. The frozen `prompt_text` is the model input; references and executable tests are consumed only by CPU scorers. Every sampled trajectory contributes to the accuracy denominator. Fixed seeds and large K support statistical reproduction; arithmetic order and batch shapes can change individual trajectories.

The visual panel contains 26,668 inputs from seven sources. The report presents six of them; OCRBench is included as an additional evaluation. It uses K1, reference arithmetic, B128, temperature 1, top-k 20, top-p 0.95, presence penalty 1.5, seed 20260912 and a 32,768-token cap. One damaged AI2D item is excluded by the supplied reference correction, leaving 26,667 scored responses. Image processing uses the included visual frontend and a 65,536–4,194,304 pixel range.

Visual media are stored losslessly in `evaluation_data/vision/media.zip` and read directly by the evaluator; extraction is not required.

The memory panel contains only the four tasks in Section 9 of the technical report: exact retrieval, state-update interpretation, cross-document combination and variable binding. Each task has 64 contexts, for 256 contexts and 1,024 queries in total. Contexts form original/counterfactual pairs and span approximately 16K–130K tokens. Evaluation uses greedy K1, reference arithmetic, B1, seed 2026091707 and a 1,024-token output cap. Results include query accuracy, all-four-correct accuracy, the four task breakdowns, and both-correct accuracy for paired queries whose answers should change or remain stable.

### Scoring

All 29 text sources use the v41 answer-verification implementation in `runtime/evaluation/` and `runtime/scoring/`. Final-answer extraction is independent of the reference answer; it recognizes explicit terminal choices and equivalent numeric forms while retaining ambiguity checks. Each benchmark keeps its scoring metric and full questions × K denominator. `RESULTS.json` records `cpu_verifier_revision`.

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

`GENERATION_COMPLETE.json` records persisted generation counts. `RESULTS.json` contains scores, denominators and sampling settings after scoring. A partial run is marked `INCOMPLETE`. Text diagnostics include protocol pass rate, answer visibility, EOS and cap-hit rates, mechanical loops, length distributions and correct answers per million output tokens.

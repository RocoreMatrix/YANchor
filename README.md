<div align="center">
  <h1>YANchor-4B</h1>
  <h3>Effective Long-Horizon Reasoning<br>in O(N) Time with O(1) Memory</h3>
  <p>Rocore Matrix · October 2026 · arXiv:2610.10118</p>
  <p>
    📄 <a href="https://arxiv.org/abs/2610.10118"><b>Technical Report</b></a>
    &nbsp; | &nbsp;
    💻 <a href="https://github.com/RocoreMatrix/YANchor">GitHub</a>
    &nbsp; | &nbsp;
    🤗 <a href="https://huggingface.co/HuishanJi/YANchor-4B">Hugging Face</a>
    &nbsp; | &nbsp;
    🤖 <a href="https://modelscope.ai/models/RocoreMatrix/YANchor-4B">ModelScope</a>
  </p>
</div>

YANchor-4B is a general-purpose recurrent model that preserves crucial memory as ANchors for retrieval during subsequent reasoning. Its multidimensional memory mechanism enables effective long-horizon reasoning with O(N) generation time and O(1) history-state memory.

For example, on challenging math problems, YANchor achieves 82.93% mean pass@1 on AIME 2024–2026 and 63.64% on HMMT, substantially outperforming linear-time, constant-state counterparts, including 7B–14B models.

YANchor delivers several-fold higher batched long-generation throughput than Transformer and hybrid baselines on H100. Evaluations across dozens of benchmarks demonstrate its superiority in general-purpose capabilities.

![Capability comparison](assets/capability-comparison.png)

AIME averages 2024–2026; HMMT uses February 2026. Code averages HumanEval, MBPP, and LiveCodeBench v6; science averages GPQA-Diamond, ARC-C, and ARC-E. Benchmarks have equal weight within each aggregate. [Evaluation settings](#evaluation-settings).

**Single-H100 inference: speed, throughput, and memory.**

![Single-H100 inference overview](assets/inference-overview.png)

Throughput uses repeated prompts and 32K outputs. Solid memory curves use measured components; dashed curves project KV growth. [Inference settings](#inference-configurations) · [Memory comparison](#gpu-memory).

## Evaluation

Evaluation covers 27 text benchmarks and six VL benchmarks. YANchor reports mean pass@1 for correctness-based tasks; Evaluation Settings defines the metrics. Bold marks row-best scores, including ties.

The tables below reproduce the technical report’s measurements. The current downloadable evaluation data covers complete designated splits; results on this expanded data release have not yet been measured.

### Linear-time, constant-state models

YANchor leads all 16 evaluated linear-time, constant-state baselines on 20 of the 24 common benchmarks. Its 24-benchmark mean is **78.64**, exceeding the strongest baseline, RWKV-7 G1j 13.3B at 56.29, by **22.35 points**. On AIME and HMMT, YANchor scores 82.93 and 63.64, compared with 18.89 and 12.12 for that 13.3B model. The lead spans 7B–14B counterparts and extends to knowledge, code, and instruction following.

| Benchmark | YANchor-4B | ARWKV-R1 7B [[2]](#ref-2) | Falcon3-Mamba 7B Instruct [[4]](#ref-4) | QRWKV7 7B Instruct [[5]](#ref-5) | xLSTM 7B [[6]](#ref-6) | Recurrent Gemma 9B IT [[8]](#ref-8) | RWKV-7 G1j 13.3B [[7]](#ref-7) | AHN-GDN Qwen2.5 14B [[3]](#ref-3) | Phi-3 Medium 4K Instruct W2047 [[9]](#ref-9) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mathematics** |  |  |  |  |  |  |  |  |  |
| AIME 2024 [[12]](#ref-12) | **86.72** | 0.00 | 1.67 | 0.00 | 0.00 | 0.42 | 17.50 | 13.75 | 3.33 |
| AIME 2025 [[12]](#ref-12) | **77.45** | 0.00 | 0.83 | 0.00 | 0.00 | 0.42 | 27.92 | 10.00 | 3.33 |
| AIME 2026 [[12]](#ref-12) | **84.64** | 0.42 | 0.83 | 0.00 | 0.00 | 0.00 | 11.25 | 10.42 | 0.00 |
| HMMT Feb 2026 [[13]](#ref-13) | **63.64** | 0.76 | 2.27 | 0.00 | 1.52 | 1.89 | 12.12 | 4.92 | 3.03 |
| MATH-500 [[11]](#ref-11) | **97.49** | 31.00 | 41.20 | 41.00 | 24.80 | 22.60 | 77.00 | 80.00 | 49.00 |
| GSM8K [[10]](#ref-10) | **94.45** | 44.50 | 60.60 | 65.00 | 41.60 | 51.80 | 94.30 | 93.10 | 84.60 |
| **Knowledge** |  |  |  |  |  |  |  |  |  |
| MMLU [[14]](#ref-14) | **87.87** | 36.10 | 61.50 | 66.20 | 23.50 | 56.00 | 76.80 | 77.30 | 75.90 |
| MMLU-Pro [[15]](#ref-15) | **79.60** | 13.30 | 31.70 | 27.60 | 21.00 | 19.90 | 60.60 | 64.70 | 58.80 |
| MMLU-Redux [[16]](#ref-16) | **92.00** | 36.69 | 64.26 | 35.32 | 38.99 | 54.82 | 79.66 | 83.33 | 81.45 |
| CMMLU [[17]](#ref-17) | 72.40 | 29.47 | 35.28 | 57.90 | 19.35 | 40.42 | 66.14 | **75.78** | 53.60 |
| C-Eval [[18]](#ref-18) | 74.47 | 28.60 | 39.00 | 57.20 | 19.20 | 40.70 | 66.60 | **76.40** | 50.90 |
| **Science** |  |  |  |  |  |  |  |  |  |
| GPQA-Diamond [[19]](#ref-19) | **64.61** | 13.64 | 24.24 | 10.61 | 12.63 | 25.25 | 42.42 | 40.91 | 29.80 |
| SuperGPQA [[20]](#ref-20) | **61.69** | 9.90 | 12.30 | 16.90 | 5.90 | 19.30 | 35.50 | 36.30 | 25.30 |
| ARC-C [[23]](#ref-23) | **95.50** | 50.30 | 76.70 | 86.00 | 27.10 | 73.20 | 91.60 | 90.90 | 90.20 |
| ARC-E [[23]](#ref-23) | **98.02** | 58.10 | 86.90 | 96.00 | 29.00 | 87.40 | 96.60 | 95.10 | 95.50 |
| **Commonsense and reasoning** |  |  |  |  |  |  |  |  |  |
| WinoGrande [[21]](#ref-21) | 77.90 | 36.50 | 59.90 | 61.80 | 25.80 | 56.40 | 72.50 | 77.20 | **80.60** |
| HellaSwag [[22]](#ref-22) | 64.57 | 12.70 | 41.00 | 45.20 | 9.20 | 44.90 | **76.40** | 70.00 | 76.10 |
| BBH [[24]](#ref-24) | **85.88** | 12.00 | 31.50 | 37.39 | 10.90 | 35.30 | 48.89 | 53.57 | 51.87 |
| BBEH [[25]](#ref-25) | **29.07** | 2.11 | 3.51 | 1.91 | 3.32 | 2.41 | 7.64 | 4.50 | 2.61 |
| DROP F1 [[31]](#ref-31) | **77.22** | 10.38 | 19.87 | 21.02 | 30.71 | 38.25 | 54.38 | 65.32 | 50.30 |
| MuSR [[32]](#ref-32) | **60.42** | 10.96 | 23.34 | 1.84 | 10.92 | 35.04 | 51.35 | 56.13 | 50.55 |
| LogiQA2 [[33]](#ref-33) | **72.23** | 21.10 | 40.20 | 26.10 | 18.00 | 40.40 | 66.70 | 70.70 | 61.50 |
| **Code** |  |  |  |  |  |  |  |  |  |
| HumanEval [[26]](#ref-26) | **96.72** | 16.46 | 34.15 | 61.59 | 36.59 | 37.20 | 85.98 | 75.61 | 73.78 |
| MBPP [[27]](#ref-27) | **87.81** | 21.20 | 39.20 | 30.20 | 9.20 | 19.00 | 67.40 | 68.60 | 59.40 |
| LiveCodeBench v6 [[28]](#ref-28) | **61.63** | 2.60 | 9.50 | 12.70 | 2.30 | 8.20 | 31.00 | 33.30 | 19.60 |
| **Instruction following** |  |  |  |  |  |  |  |  |  |
| IFEval [[29]](#ref-29) | **88.03** | 10.35 | 64.70 | 58.78 | 35.49 | 39.56 | 77.82 | 77.26 | 61.55 |
| IFBench [[30]](#ref-30) | **65.29** | 8.33 | 18.67 | 21.33 | 14.67 | 12.67 | 27.33 | 33.33 | 22.00 |

## Model

| Model | Text parameters | Weights | Model license |
|---|---:|---|---|
| YANchor-4B | 4.58B | [Hugging Face](https://huggingface.co/HuishanJi/YANchor-4B) | [YANchor Model License 1.0](MODEL_LICENSE) |

### Architecture

![YANchor architecture](assets/architecture.png)

Long-horizon reasoning needs access to crucial information after it leaves the local context. YANchor preserves this information as persistent memory anchors, making distant context available for retrieval during subsequent reasoning.

YANchor retains Qwen3.5-4B's recurrent layers and replaces its global-attention positions with local attention and independent memory. Three complementary paths carry earlier constraints, intermediate results, and exact associations.

| Information path | Role |
|---|---|
| Gated DeltaNet | Maintains a fixed-size recurrent summary of history. |
| Sliding-window attention | Resolves detailed interactions among recent tokens. |
| Independent GQA memory | Learns what to write, which records to retain, and how to retrieve distant information. |

The memory has independent writing, admission, and retrieval parameters. New records compete with retained entries through blockwise Top-K selection, which preserves the written key–value representations of the retained records. Current queries read these memory anchors through grouped-query attention. Local and memory attention are normalized separately, and their outputs join the residual stream.

The recurrent state, local window, memory bank, candidate records, and metadata have fixed capacity. YANchor therefore generates N tokens in O(N) time using O(1) history-state memory.

## Inference efficiency

Fixed-capacity history supports sustained decoding and larger active batches. All comparisons use one H100 80GB.

**Single-sequence speed.** YANchor sustains 212.98 cumulative decode tokens/s at 128K input + 128K output; Qwen3.5-4B, Gemma 4 E4B, and MiniCPM5-2B reach 153.91, 133.75, and 164.53 tokens/s, respectively. YANchor's curve remains nearly flat as generation extends.

**Batched throughput.** YANchor delivers 5.66–6.58 times the throughput of Gemma, MiniCPM, and Qwen at 32K input + 32K output. It reaches 12,431 end-to-end tokens/s, compared with 2,196, 2,117, and 1,889, respectively.

![Single-sequence speed and batched throughput](assets/inference-efficiency.png)

### Inference configurations

Tests use synthetic token inputs and fixed-length generation with EOS ignored; tokenizer and text-decoding costs are excluded. Batch tests use repeated prompts with independent continuation states. Batch candidates are 32–512 for YANchor, 64/128 for Gemma, and 32/64 for Qwen and MiniCPM; the comparison reports each model's highest measured throughput. Tensor-parallel size is one. Pure decode is timed after first-token delivery; end-to-end throughput covers the complete generation call, including prefill, queueing, scheduling, and cache-preemption recomputation. Loading and warmup are outside both intervals. Here 1K is 1,024 tokens.

YANchor uses CUDA Graphs and fused kernels, with INT8 MLP / FP32 recurrent state at B1 and BF16 MLP / FP16 recurrent state in batch tests. Qwen uses BF16, FlashAttention-4, FlashInfer GDN prefill, and fused CUDA GDN decode. Gemma uses BF16 and full-layer FlashAttention-4; MiniCPM uses BF16 and FlashAttention-3. All three baselines enable vLLM O3, CUDA Graphs, chunked prefill, and asynchronous scheduling.

Gemma and MiniCPM extend runtime and position-table capacity to 256K for the B1 speed measurement, keeping weights and RoPE frequencies unchanged. Deployment references: [Gemma/vLLM](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html), [MiniCPM5](https://huggingface.co/openbmb/MiniCPM5-2B#quickstart).

## GPU memory

YANchor keeps 512 independent long-context sequences resident on one H100 80GB. Its history cache stays fixed at 48.90 GiB as context grows from 512 to 65,536 tokens per sequence. Peak memory demand remains near 67 GiB; the 1 GiB increase comes from input buffers.

![GPU memory versus context length](assets/gpu-memory.png)

Each sequence retains an independent history. Solid curves reconstruct peak demand from measured occupied cache, workspace, and runtime overhead, excluding unused preallocated cache. Dashed curves project full-batch demand from the last successful point. The device has 79.65 GiB of memory.

Cache preemption begins at 4K context for Qwen and MiniCPM, and at 8K for Gemma. Their projected demands at 64K are 1,061, 1,352, and 541 GiB, respectively. YANchor's fixed history storage enables the longer contexts and larger active batches that drive its throughput.

## Long-memory capability

YANchor retrieves and uses distant information with 94.14% query accuracy over 16K–130,000-token contexts. The tasks cover exact retrieval, state updates, cross-document combination, and variable binding.

| Memory path | Query accuracy | All four answers correct |
|---|---:|---:|
| Enabled | 94.14% | 80.08% |
| Disabled | 0.39% | 0.00% |

Disabling memory reduces query accuracy to 0.39%. Paired evidence-change tests further show that YANchor uses distant content to update its answers. The technical report gives the task breakdown and experimental settings.

## Transformer and global-attention hybrid references

These models provide cross-architecture references with context-dependent history storage. Qwen3.5-4B combines Gated DeltaNet with global attention and serves as the backbone reference for YANchor’s conversion to bounded-state inference.

YANchor surpasses MiniCPM4.1-8B, Gemma4-E4B, Phi-4-mini-reasoning, and Nemotron-3-Nano-4B on all three AIME editions while using constant history-state memory. Compared with Qwen3.5-4B, YANchor scores higher on MMLU, MMLU-Pro, MMLU-Redux, SuperGPQA, HumanEval, MBPP, and IFBench. Across the 24 common benchmarks, YANchor and Qwen score 78.64 and 80.44, respectively. YANchor retains 96.31% of Qwen’s three-year AIME mean while using 43–47% fewer output tokens, and delivers 4.12–6.58 times its measured end-to-end throughput in batched long generation.

| Benchmark | YANchor-4B | Qwen3.5-4B Backbone [[1]](#ref-1) | Granite4.2 3B [[41]](#ref-41) | Phi-4-mini-reasoning [[42]](#ref-42) | Gemma4 E4B-it [[40]](#ref-40) | Nemotron-3 Nano-4B [[43]](#ref-43) | MiniCPM4.1 8B [[39]](#ref-39) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mathematics** |  |  |  |  |  |  |  |
| AIME 2024 [[12]](#ref-12) | 86.72 | **89.58** | 85.00 | 44.58 | 41.25 | 60.83 | 77.08 |
| AIME 2025 [[12]](#ref-12) | 77.45 | **80.83** | 78.75 | 30.83 | 39.58 | 56.67 | 70.00 |
| AIME 2026 [[12]](#ref-12) | 84.64 | **87.92** | 82.92 | 35.00 | 38.33 | 53.75 | 70.83 |
| HMMT Feb 2026 [[13]](#ref-13) | 63.64 | **65.91** | 58.71 | 26.14 | 28.03 | 36.36 | 50.00 |
| MATH-500 [[11]](#ref-11) | 97.49 | **98.00** | 97.80 | 91.40 | 87.40 | 94.80 | 96.40 |
| GSM8K [[10]](#ref-10) | 94.45 | **94.50** | 93.50 | 92.40 | 90.30 | 91.80 | 93.30 |
| **Knowledge** |  |  |  |  |  |  |  |
| MMLU [[14]](#ref-14) | **87.87** | 85.60 | 77.30 | 78.10 | 75.20 | 77.30 | 82.20 |
| MMLU-Pro [[15]](#ref-15) | **79.60** | 75.90 | 65.80 | 53.20 | 68.60 | 56.40 | 65.90 |
| MMLU-Redux [[16]](#ref-16) | **92.00** | 89.31 | 80.92 | 79.77 | 85.95 | 80.82 | 86.79 |
| CMMLU [[17]](#ref-17) | 72.40 | **82.26** | 58.29 | 44.34 | 61.84 | 53.94 | 80.12 |
| C-Eval [[18]](#ref-18) | 74.47 | **85.10** | 61.80 | 42.70 | 64.90 | 55.70 | 78.80 |
| **Science** |  |  |  |  |  |  |  |
| GPQA-Diamond [[19]](#ref-19) | 64.61 | **77.78** | 58.59 | 47.98 | 40.91 | 43.43 | 50.51 |
| SuperGPQA [[20]](#ref-20) | **61.69** | 51.40 | 39.60 | 34.70 | 42.50 | 37.50 | 41.90 |
| ARC-C [[23]](#ref-23) | 95.50 | **96.90** | 92.80 | 91.40 | 92.10 | 91.60 | 92.70 |
| ARC-E [[23]](#ref-23) | 98.02 | **99.00** | 97.30 | 95.70 | 96.60 | 96.10 | 96.50 |
| **Commonsense and reasoning** |  |  |  |  |  |  |  |
| WinoGrande [[21]](#ref-21) | 77.90 | **87.00** | 74.60 | 75.90 | 76.40 | 65.30 | 72.70 |
| HellaSwag [[22]](#ref-22) | 64.57 | **80.10** | 67.70 | 62.60 | 52.20 | 46.70 | 46.50 |
| BBH [[24]](#ref-24) | 85.88 | 85.81 | 86.22 | 74.92 | 85.10 | 80.00 | **88.10** |
| BBEH [[25]](#ref-25) | 29.07 | **34.31** | 25.10 | 12.26 | 16.60 | 9.22 | 12.79 |
| DROP F1 [[31]](#ref-31) | 77.22 | 75.41 | **83.35** | 54.05 | 72.58 | 71.95 | 78.60 |
| MuSR [[32]](#ref-32) | 60.42 | 64.78 | **65.07** | 52.45 | 62.18 | 58.24 | 64.49 |
| LogiQA2 [[33]](#ref-33) | 72.23 | **81.30** | 68.70 | 57.60 | 73.50 | 67.30 | 76.80 |
| **Code** |  |  |  |  |  |  |  |
| HumanEval [[26]](#ref-26) | 96.72 | 85.98 | 96.95 | 81.71 | 93.90 | 90.85 | **98.17** |
| MBPP [[27]](#ref-27) | 87.81 | 70.80 | **92.60** | 64.00 | 79.20 | 72.20 | 92.20 |
| LiveCodeBench v6 [[28]](#ref-28) | 61.63 | 76.20 | **76.30** | 35.80 | 69.50 | 67.10 | 70.80 |
| **Instruction following** |  |  |  |  |  |  |  |
| IFEval [[29]](#ref-29) | 88.03 | 90.02 | **92.42** | 46.03 | 84.29 | 89.28 | 73.75 |
| IFBench [[30]](#ref-30) | 65.29 | 60.33 | **74.33** | 14.67 | 34.67 | 57.67 | 23.67 |

## VL comparison

YANchor supports bilingual VL question answering, object-presence judgments, visual perception, chart understanding, and scientific-diagram reasoning.

| Benchmark | YANchor-4B | Qwen3.5-4B [[1]](#ref-1) |
| --- | ---: | ---: |
| **Vision-language** |  |  |
| MMBench EN [[34]](#ref-34) | 88.28 | **90.08** |
| MMBench CN [[34]](#ref-34) | 89.00 | **91.43** |
| POPE [[35]](#ref-35) | 85.43 | **88.76** |
| MME [[36]](#ref-36) | 82.98 | **86.82** |
| ChartQA [[37]](#ref-37) | 83.92 | **90.28** |
| AI2D [[38]](#ref-38) | 80.05 | **86.98** |

VL scores are answer-level accuracy in percent.

## Evaluation Settings

**YANchor mean pass@1.** For text evaluation, YANchor sets the number of independent responses per item, K, by evaluation-set size: K=64 for fewer than 100 questions, K=16 for 100–10,000 questions, and K=1 for more than 10,000 questions. More repetitions on smaller sets reduce variation from stochastic generation. Binary-correctness scores average across responses and then across problems, estimating single-response success. Other metrics retain their benchmark-specific definitions.

**Scoring.** Mathematics uses final-answer correctness. YANchor's code scores count extracted programs that pass the task tests, including code recovered from the end of reasoning. Knowledge and instruction-following tasks use their benchmark-specific aggregation and checkers.

### Generation settings

YANchor's text evaluation uses temperature 0.6, top-p 0.95, top-k 20, and a 131,072-token response budget. Completions beyond this budget receive zero outcome credit. VL evaluation uses one response per input with thinking enabled, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, and a 32,768-token response cap.

## Quickstart

### Environment

The released runtime is validated on **NVIDIA H100 80 GB with Linux**. Use Docker with NVIDIA Container Toolkit and the supplied `Dockerfile`, based on `nvcr.io/nvidia/pytorch:26.03-py3`. The image supplies PyTorch, CUDA and FlashAttention; the remaining dependencies, including the pinned public Transformers wheel, are included or specified in the bundle.

YANchor-4B uses the custom inference runtime provided through **`run.py`**. Use this entrypoint to load the model and its memory modules.

### Download and run

Download the complete [Hugging Face bundle](https://huggingface.co/HuishanJi/YANchor-4B), which includes the model weights, runtime and evaluation data. The bundle is also available on [ModelScope](https://modelscope.ai/models/RocoreMatrix/YANchor-4B).

```bash
python -m pip install -U huggingface_hub
hf download HuishanJi/YANchor-4B --local-dir YANchor-4B
cd YANchor-4B
docker build -t yanchor-runtime .
docker run --rm --gpus all --ipc=host -v "$PWD:/model" \
  yanchor-runtime demo --prompt 'Compute 17 × 23.'
```

Run the commands below from the downloaded bundle root. First startup includes weight loading and CUDA compilation. See [the usage guide](USAGE.md) for dependency installation and additional examples.

## Deployment

Start a local B1 HTTP service with the fastest released inference profile:

```bash
docker run --rm --gpus all --ipc=host --network host -v "$PWD:/model" \
  yanchor-runtime serve --batch 1 --profile fast --port 8000
```

The service prewarms its short-input decode path before reporting READY. Batch generation uses the same runtime:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/model" \
  yanchor-runtime generate --batch 320 --profile fast \
  --input prompts.jsonl --output outputs/generated
```

`prompts.jsonl` contains one `{"prompt":"..."}` object per line. Supported per-GPU batch capacities are **1, 32, 64, 128, 256, 320 and 512**. The `fast` profile uses INT8 MLP GEMV with FP32 GDN state for B1, and BF16 MLPs with FP16 GDN state for B32–B512. `--profile reference` selects the reference arithmetic. Weights on disk remain BF16. Model loading, prefill and pure decode timings are reported separately.

See [deployment and benchmarking](USAGE.md#demo-and-batch-inference) for request formats, all batch configurations and fixed-work throughput measurements.

## Running the evaluations

The complete model bundle includes all **113,743 questions across the designated evaluation splits of 29 text benchmarks**, together with the scoring code. By default, the evaluator runs every question in each source, with no per-source question limit. The following command uses one H100; expand `--gpus` to a comma-separated list to use more GPUs:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD:/model" \
  yanchor-runtime evaluate --suite text --gpus 0 --batch 320 --output outputs/text
```

Automatic K uses each bundled source's question count: **K=64** below 100 questions, **K=16** for 100–10,000, and **K=1** above 10,000. The defaults apply the text generation settings above and the **v41** scoring rules. Source scores use all question × K attempts and retain benchmark-specific aggregation.

Monitor progress and aggregate decode throughput from another terminal:

```bash
docker run --rm -v "$PWD:/model" yanchor-runtime status --output outputs/text
```

See [evaluation instructions](USAGE.md#evaluation) for text, vision and the four reported memory tasks, source selection, scoring rules and result files. The current full-split release expands the evaluation data. The benchmark tables retain the technical report’s original measurements; they have not been recomputed on this expanded release.

## License

YANchor-4B weights use [YANchor Model License 1.0](MODEL_LICENSE), a custom license incorporating Apache License 2.0 terms. The sole additional restriction is physically hardwiring covered weights into a chip's non-reprogrammable structure, including commissioning such fabrication, without written authorization. Research, training, fine-tuning, distillation, quantization, deployment, redistribution and commercial use are permitted without fees or scale thresholds. Ordinary device deployment, model-data storage and hardware design remain permitted; read-only storage alone does not trigger the restriction. Code uses [Apache-2.0](LICENSE). Upstream and third-party materials retain their original licenses; see [NOTICE](NOTICE).

## Citation

```bibtex
@misc{yanchor2026,
  title  = {{YANchor-4B}: Effective Long-Horizon Reasoning in {O(N)} Time with {O(1)} Memory},
  author = {Ji, Huishan and Xu, Hua and Zhang, Weiming and Ye, Qirui},
  year   = {2026},
  eprint = {2610.10118},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2610.10118}
}
```

Please cite the [technical report, arXiv:2610.10118](https://arxiv.org/abs/2610.10118). Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

### Contacts

Huishan Ji · [jihs@stonehill-tech.com](mailto:jihs@stonehill-tech.com)

Hua Xu · [xuh@stonehill-tech.com](mailto:xuh@stonehill-tech.com)

Weiming Zhang · [zhangwm@stonehill-tech.com](mailto:zhangwm@stonehill-tech.com)

Qirui Ye · [qiruiy@andrew.cmu.edu](mailto:qiruiy@andrew.cmu.edu)

## Acknowledgments

YANchor builds on [Qwen3.5](https://github.com/QwenLM/Qwen3.5) and [Gated DeltaNet](https://arxiv.org/abs/2412.06464). We thank the teams behind the comparison models and evaluation benchmarks.

## References

<a id="ref-1"></a>
[1] Qwen Team. Qwen3.5-4B. Model card. <https://huggingface.co/Qwen/Qwen3.5-4B>, 2026.

<a id="ref-2"></a>
[2] RWKV Red Team. ARWKV-R1-7B. Model release. <https://huggingface.co/RWKV-Red-Team/ARWKV-R1-7B>, 2025.

<a id="ref-3"></a>
[3] Y. Fang et al. Artificial Hippocampus Networks for Efficient Long-Context Modeling. <https://arxiv.org/abs/2510.07318>, 2025.

<a id="ref-4"></a>
[4] Technology Innovation Institute. Falcon3-Mamba-7B-Instruct. <https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Instruct>, 2024.

<a id="ref-5"></a>
[5] D. Goldstein, E. Alcaide, J. Lu, and E. Cheah. RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale. <https://arxiv.org/abs/2505.03005>, 2025.

<a id="ref-6"></a>
[6] M. Beck et al. xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference. <https://arxiv.org/abs/2503.13427>, 2025.

<a id="ref-7"></a>
[7] RWKV Project. RWKV-7 G1 model releases. <https://huggingface.co/BlinkDL/rwkv7-g1>, 2026.

<a id="ref-8"></a>
[8] A. Botev et al. RecurrentGemma: Moving Past Transformers for Efficient Open Language Models. <https://arxiv.org/abs/2404.07839>, 2024.

<a id="ref-9"></a>
[9] Microsoft. Phi-3 Medium 4K Instruct. Released model configuration. <https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json>, 2024.

<a id="ref-10"></a>
[10] K. Cobbe et al. Training Verifiers to Solve Math Word Problems. <https://arxiv.org/abs/2110.14168>, 2021.

<a id="ref-11"></a>
[11] H. Lightman et al. Let's Verify Step by Step. <https://arxiv.org/abs/2305.20050>, 2023.

<a id="ref-12"></a>
[12] Mathematical Association of America. American Invitational Mathematics Examination. <https://maa.org/maa-invitational-competitions/>.

<a id="ref-13"></a>
[13] Harvard-MIT Mathematics Tournament. Past Tournaments. <https://www.hmmt.org/www/archive/results>.

<a id="ref-14"></a>
[14] D. Hendrycks et al. Measuring Massive Multitask Language Understanding. <https://arxiv.org/abs/2009.03300>, 2020.

<a id="ref-15"></a>
[15] Y. Wang et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. <https://arxiv.org/abs/2406.01574>, 2024.

<a id="ref-16"></a>
[16] A. P. Gema et al. Are We Done with MMLU? <https://arxiv.org/abs/2406.04127>, 2024.

<a id="ref-17"></a>
[17] H. Li et al. CMMLU: Measuring Massive Multitask Language Understanding in Chinese. <https://arxiv.org/abs/2306.09212>, 2023.

<a id="ref-18"></a>
[18] Y. Huang et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. <https://arxiv.org/abs/2305.08322>, 2023.

<a id="ref-19"></a>
[19] D. Rein et al. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. <https://arxiv.org/abs/2311.12022>, 2023.

<a id="ref-20"></a>
[20] M-A-P Team et al. SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines. <https://arxiv.org/abs/2502.14739>, 2025.

<a id="ref-21"></a>
[21] K. Sakaguchi et al. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. <https://arxiv.org/abs/1907.10641>, 2019.

<a id="ref-22"></a>
[22] R. Zellers et al. HellaSwag: Can a Machine Really Finish Your Sentence? <https://arxiv.org/abs/1905.07830>, 2019.

<a id="ref-23"></a>
[23] P. Clark et al. Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. <https://arxiv.org/abs/1803.05457>, 2018.

<a id="ref-24"></a>
[24] M. Suzgun et al. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. <https://arxiv.org/abs/2210.09261>, 2022.

<a id="ref-25"></a>
[25] M. Kazemi et al. BIG-Bench Extra Hard. <https://arxiv.org/abs/2502.19187>, 2025.

<a id="ref-26"></a>
[26] M. Chen et al. Evaluating Large Language Models Trained on Code. <https://arxiv.org/abs/2107.03374>, 2021.

<a id="ref-27"></a>
[27] J. Austin et al. Program Synthesis with Large Language Models. <https://arxiv.org/abs/2108.07732>, 2021.

<a id="ref-28"></a>
[28] N. Jain et al. LiveCodeBench. Benchmark repository, release v6. <https://github.com/LiveCodeBench/LiveCodeBench>, 2025.

<a id="ref-29"></a>
[29] J. Zhou et al. Instruction-Following Evaluation for Large Language Models. <https://arxiv.org/abs/2311.07911>, 2023.

<a id="ref-30"></a>
[30] V. Pyatkin et al. Generalizing Verifiable Instruction Following. <https://arxiv.org/abs/2507.02833>, 2025.

<a id="ref-31"></a>
[31] D. Dua et al. DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. <https://arxiv.org/abs/1903.00161>, 2019.

<a id="ref-32"></a>
[32] Z. Sprague et al. MuSR: Testing the Limits of Chain-of-Thought with Multistep Soft Reasoning. <https://arxiv.org/abs/2310.16049>, 2023.

<a id="ref-33"></a>
[33] H. Liu et al. LogiQA 2.0: An Improved Dataset for Logical Reasoning in Natural Language Understanding. <https://github.com/csitfun/LogiQA2.0>, 2023.

<a id="ref-34"></a>
[34] Y. Liu et al. MMBench: Is Your Multi-modal Model an All-around Player? <https://arxiv.org/abs/2307.06281>, 2023.

<a id="ref-35"></a>
[35] Y. Li et al. Evaluating Object Hallucination in Large Vision-Language Models. <https://arxiv.org/abs/2305.10355>, 2023.

<a id="ref-36"></a>
[36] C. Fu et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. <https://arxiv.org/abs/2306.13394>, 2023.

<a id="ref-37"></a>
[37] A. Masry et al. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. <https://arxiv.org/abs/2203.10244>, 2022.

<a id="ref-38"></a>
[38] A. Kembhavi et al. A Diagram Is Worth a Dozen Images. <https://arxiv.org/abs/1603.07396>, 2016.

<a id="ref-39"></a>
[39] OpenBMB. MiniCPM4.1-8B. Model card. <https://huggingface.co/openbmb/MiniCPM4.1-8B>, 2025.

<a id="ref-40"></a>
[40] Google. Gemma 4 E4B. Model card. <https://huggingface.co/google/gemma-4-E4B-it>, 2026.

<a id="ref-41"></a>
[41] IBM. Granite 4.2 3B. Model card. <https://huggingface.co/ibm-granite/granite-4.2-3b>, 2026.

<a id="ref-42"></a>
[42] Microsoft. Phi-4-mini-reasoning. Model card. <https://huggingface.co/microsoft/Phi-4-mini-reasoning>, 2025.

<a id="ref-43"></a>
[43] NVIDIA. Nemotron 3 Nano 4B. Model card. <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16>, 2026.

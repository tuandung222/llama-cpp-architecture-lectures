# AGENT.md

## 1. Project overview

This repository is a Vietnamese Docusaurus curriculum for the **llama.cpp** library, the most popular open-source LLM inference engine. It is designed for deep learning engineers, AI researchers, model serving developers, and systems engineers who want to understand the exact mathematical foundations, codebase implementation, and performance tuning strategies of efficient LLM inference.

The goal is not simply to copy tutorials. The goal is to provide a rigorous, academic-grade guide explaining the theoretical roots (quantization error math, tensor layout algebra, attention optimization, KV Cache memory analysis) and connecting them to corresponding codebase implementations in llama.cpp and GGML, backed by clear pseudocode and systems analysis.

This file is the operating manual for future agents. Treat it as the stable source of truth for writing quality, repository safety, verification, privacy, and completion criteria.

---

## 2. Repository map

- `docs/`: public Vietnamese curriculum chapters.
  - `docs/case_studies/`: practical case studies (quant comparison, model porting, edge deployment, speculative decoding, LoRA conversion).
  - `docs/theory_deep_dive/`: theoretical, mathematical, and pseudocode deep dives.
  - `docs/experiments_deep_dive/`: hands-on labs (build, benchmark, deploy, evaluate).
- `src/`: Docusaurus landing page and styling.
- `static/`: public static assets and `robots.txt`.
- `.github/workflows/`: CI and GitHub Pages deployment workflow.
- `README.md`: must remain empty.

---

## 3. Curriculum-wide content standard

Public content must be highly educational, mathematically rigorous, and written for advanced learners. Do not expose private task instructions, local absolute paths, credentials, internal notes, hidden constraints, or agent coordination details in public docs.

A curriculum chapter should teach by introducing a concrete tension first (e.g., why CPU inference matters, why quantization loses quality, why KV Cache eats memory) before offering the solution.

Use `Phần` for course sections. Do not use em dash characters. Use commas, colons, semicolons, or parentheses instead.

---

## 4. Pedagogical writing style

Future agents must write with the persona of an **AI Expert, Deep Learning & LLM Inference Specialist**, and a dedicated professor. The goal is to help students grasp the absolute roots, mathematical foundations, and system mechanics of efficient LLM inference, rather than just high-level summaries, so they can confidently apply and implement them in real-world systems.

The prose must be highly professional, precise, serious, patient, and technically deep. It must read like an original academic lecture series in Vietnamese, not a translated or marketing-oriented document.

Use Vietnamese as the main language. Use English technical terms when they are standard in the industry: *tensor, quantization, inference, attention, KV Cache, GGUF, GGML, SIMD, AVX2, NEON, CUDA, Metal, Vulkan, RoPE, flash attention, speculative decoding, LoRA, perplexity, sampling, batch processing*. Explain a term before relying on it heavily.

Avoid casual language, slang, and jokes. The tone should be authoritative, academic, and accessible.

For every important concept, follow this pedagogical flow:
1. Start from a concrete hardware/system tension (e.g., memory bandwidth bottleneck, GPU underutilization, model too large for RAM).
2. Build mathematical intuition, formulate equations (using LaTeX), and prove key terms.
3. Show the corresponding clean pseudocode or algorithms.
4. Reference the actual file and function in llama.cpp where this is implemented.
5. Provide actionable performance tuning checklists and practical implementation guides.

---

## 5. Math, diagrams, and examples

Math must be taught, not just displayed.
- Explain every variable in equations.
- Use LaTeX formatting ($formula$ or $$formula$$).
- Follow equations with intuitive prose, such as: `Đọc công thức này theo nghĩa toán học và thực tế...` or `Bản chất của công thức nằm ở việc...`.

Use Mermaid diagrams to illustrate data flow, tensor computation graphs, inference pipeline, quantization block layout, and KV Cache management.

---

## 6. Public privacy and safety constraints

`README.md` must remain empty (0 bytes). Do not add any characters or placeholders to it.

Public docs must not mention:
- private user instructions or hidden agent constraints.
- the fact that `README.md` is empty.
- local absolute paths.
- credentials, tokens, secrets, API keys, or private URLs.

Privacy controls:
- `static/robots.txt` must disallow all crawling.
- Docusaurus must include `noindex,nofollow,noarchive,nosnippet` metadata.
- Sitemap generation must remain disabled.

---

## 7. Commands and verification

Safe read-only or verification commands:
- `npm run typecheck`: run TypeScript verification.
- `npm run build`: build the Docusaurus site.
- `git status --short --branch`: inspect repository state.
- `gh api repos/tuandung222/llama-cpp-architecture-lectures/pages`: verify GitHub Pages deployment status.

Commands requiring explicit approval/actions:
- Configuring or enabling GitHub Pages for the first time:
  `echo '{"source": {"branch": "gh-pages", "path": "/"}}' | gh api --method POST /repos/tuandung222/llama-cpp-architecture-lectures/pages --input -`
- Manual publishing or deploying.
- Pushing to GitHub if not already requested.
- Changing repository visibility.

---

## 8. Completion checklist

Before reporting completion, verify the relevant items:
- `README.md` is still 0 bytes.
- `npm run typecheck` and `npm run build` pass without errors.
- No em dash characters appear in public or source text.
- Public docs read like original Vietnamese teaching material.
- If pushed, the commit author and committer are the intended identity (`tuandung222`).
- The deployed website returns `HTTP 200` on the live URL.
- Search engine exclusions are active on the live site (verified through `robots.txt` disallowing `/` and `<meta name="robots" content="noindex..."/>` in page source).

---

## 9. Repo specialization: llama.cpp Internals

### Learning promise
A reader should finish this curriculum able to explain:
- Why llama.cpp exists and how it differs from ONNX Runtime, TensorRT-LLM, and vLLM.
- The exact structure of `ggml_tensor`, arena memory allocation, and computation graph execution in GGML.
- The binary layout of GGUF files, including header, metadata KV pairs, tensor info, and 32-byte SIMD alignment.
- The mathematical difference between Q4_0, Q4_K_M, I-quants, and their error characteristics (MSE, SQNR).
- How llama.cpp performs inference: llama_context, KV Cache management, sampling strategies, and batch processing.
- The model porting pipeline: architecture registry, weight name mapping, and convert_hf_to_gguf.py.
- How hardware backends (CPU, CUDA, Metal, Vulkan) are abstracted via ggml-backend.
- Speculative decoding mechanics and LoRA adapter integration in GGUF format.
- How BitNet b1.58 ternary weights {-1, 0, +1} enable addition-only matmul, and how bitnet.cpp kernel library (I2_S, TL1, TL2) achieves 1.7x-6.17x speedup over llama.cpp for ternary models.
- The PrismML Bonsai family (1-bit Q1_0, ternary PQ2_0/Q2_0, Bonsai 2 with a folded Hadamard basis): exact byte layouts of Q1_0, Q2_0, PQ2_0, PTQ1_0 vs TQ1_0/TQ2_0, why base-3 packing uses ceiling division, when each format is lossless, and how the `prism.hadamard.*` contract is enforced by `load_hparams_hadamard` and `build_hadamard_input`.
- How the PrismML-Eng/llama.cpp fork differs from mainline: what has been upstreamed (Q1_0, Q2_0 group 64, Hadamard runtime, MTP/DFlash/DSpark spec types) and what is fork-only (PQ2_0 = ggml type 142, PTQ1_0 = 143, backend kernels, legacy Q2_0 detection, K-cache mean-centering, GDN rollback).

### Misconceptions to actively prevent
- GGML is not a model format; it is a tensor computation library. GGUF is the format.
- Quantization does not mean the model becomes useless; Q4_K_M retains >97% quality for most models.
- CPU inference is not always slow; with AVX2/AVX512 and proper quantization, it is viable for many use cases.
- KV Cache is not free; it often exceeds model size for long contexts and must be quantized or managed.
- Speculative decoding does not change output distribution; it is a lossless speedup technique.
- LoRA adapters in llama.cpp do not require merging; they can be loaded separately with scaling control.
- BitNet b1.58 is NOT a post-training quantization method; it requires quantization-aware training with STE, normally from scratch (QAT fine-tuning of a pretrained model, e.g. Llama3-8B-1.58-100B-tokens, also works but loses quality). Simply applying absmean ternarization (or `llama-quantize ... TQ1_0`) to an existing FP16 model does not produce a usable BitNet model.
- The "1.4% vs 100%" figure (arXiv:2410.16144, Table 7) is NOT task accuracy and is NOT caused by dequantizing to float. It is the fraction of 1,000 WildChat prompts (700M model, up to 100 greedy tokens) whose output matched an FP32 reference token-for-token. llama.cpp `ggml_vec_dot_tq1_0_q8_K`/`ggml_vec_dot_tq2_0_q8_K` already compute integer dot products directly on packed trits; the mismatch comes from the scale scheme (activations quantized to `Q8_K` with one scale per 256 values instead of per-token absmax, FP16 per-block weight scale), whereas bitnet.cpp I2_S/TL1/TL2 reproduce the training-time W1.58A8 quantization exactly.
- Bonsai is not BitNet: it converts an existing pretrained model (Qwen3, Qwen3.5, Qwen3.6, Qwen3.8) and uses FP16 scales per group of 128, not per-tensor absmean.
- `llama-quantize` to PQ2_0 or PTQ1_0 does not create a Bonsai model: the reference quantizers are plain absmax rounding and are only lossless for weights that are already ternary at group 128 (and, for Bonsai 2, already in the rotated basis with `prism.hadamard.*` metadata).
- Lower bpw is not always faster: PTQ1_0 (1.75 bpw) decodes slower than PQ2_0 (2.125 bpw) on H100/A100/Blackwell because batch-1 decode there is limited by unpack arithmetic and launch overhead, not weight bandwidth.
- Mainline llama.cpp is not the same as an old llama.cpp: a runtime without the Hadamard activation transform loads Bonsai 2 `Q2_0` files silently and produces garbage.
- ggml type id and llama ftype are different numbering spaces (PQ2_0: ggml type 142, ftype 141; PTQ1_0: 143 and 143).

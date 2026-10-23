---
sidebar_position: 1
sidebar_label: "Lộ trình Lý thuyết"
---

# Lộ trình Lý thuyết & Toán học của llama.cpp

Chào mừng bạn đến với chương **Đào sâu Lý thuyết, Toán học và Giải thuật (Math & Theory Deep Dive)**.

Để làm chủ hoàn toàn hệ thống inference LLM, chúng ta phải vượt qua lớp vỏ giao diện và đi vào gốc rễ toán học của các giải thuật quantization, tensor layout, attention optimization, và evaluation metrics. Phần này cung cấp các chứng minh toán học đầy đủ, phân tích độ phức tạp, và mã giả đi kèm tham chiếu mã nguồn `llama.cpp`.

---

```mermaid
graph TD
    T1[Bài 1: Toán học Quantization Error] --> T2[Bài 2: Tensor Layout & Alignment]
    T2 --> T3[Bài 3: Attention trên CPU]
    T3 --> T4[Bài 4: KV Cache Quantization]
    T4 --> T5[Bài 5: Perplexity & Evaluation]
    T1 --> T6[Bài 6: Quantization Landscape]
    T1 --> T7[Bài 7: Ternary Quantization & BitNet]
    T6 --> T8[Bài 8: QAT vs PTQ Convergence]
    T7 --> T9[Bài 9: Định dạng GGUF của Bonsai]
    T9 --> T10[Bài 10: Fold Hadamard & FWHT]

    style T1 fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style T2 fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style T3 fill:#92400e,stroke:#f59e0b,stroke-width:2px,color:#fff
    style T4 fill:#9d174d,stroke:#ec4899,stroke-width:2px,color:#fff
    style T5 fill:#5b21b6,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style T6 fill:#dc2626,stroke:#ef4444,stroke-width:2px,color:#fff
    style T7 fill:#059669,stroke:#10b981,stroke-width:2px,color:#fff
    style T8 fill:#7c3aed,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style T9 fill:#0f766e,stroke:#14b8a6,stroke-width:2px,color:#fff
    style T10 fill:#0f766e,stroke:#14b8a6,stroke-width:2px,color:#fff
```

---

## Nội dung các bài giảng Toán học chuyên sâu

1. **[Bài 1: Toán học Quantization Error & Information Loss](theory_1_quantization_error_math)**
   * MSE analysis, SQNR, Lloyd-Max algorithm, perplexity degradation curves.
2. **[Bài 2: Đại số Tensor Layout & Memory Alignment](theory_2_tensor_layout_alignment)**
   * Stride computation, SIMD alignment math, block packing algebra.
3. **[Bài 3: Attention Mechanism trên CPU](theory_3_attention_on_cpu)**
   * KV Cache optimization, Flash Attention for CPU, RoPE mathematics.
4. **[Bài 4: KV Cache Quantization](theory_4_kv_cache_quantization)**
   * Memory footprint math, Type-0 vs Type-1, attention quality under quantization.
5. **[Bài 5: Perplexity & Evaluation Metrics](theory_5_perplexity_evaluation)**
   * Cross-entropy, PPL formula, benchmark methodology, statistical testing.
6. **[Bài 6: Cảnh giác Quantization - GGML vs GPTQ vs AWQ vs QAT](theory_6_quantization_landscape)**
   * Hessian-based OBQ, activation-aware scaling, STE, PTQ vs QAT paradigm comparison.
7. **[Bài 7: Toán học Ternary Quantization & BitNet b1.58](theory_7_ternary_quantization)**
   * Information theory log2(3), absmean quantization, addition-only matmul, scale absorption, STE, LUT mpGEMM.
8. **[Bài 8: Tại sao QAT Model chạy được trên PTQ Engine?](theory_8_qat_ptq_convergence)**
   * Phân biệt training method vs storage format vs inference engine. Pipeline BitNet QAT -> GGUF -> llama.cpp. Ý nghĩa thật của con số exact-match 1.4% của TQ so với bitnet.cpp.
9. **[Bài 9: Toán học các định dạng GGUF của Bonsai](theory_9_bonsai_gguf_formats)**
   * Bố cục byte của Q1_0, Q2_0, PQ2_0, PTQ1_0 so với TQ1_0, TQ2_0; giới hạn thông tin; chứng minh mã hóa cơ số 3 bằng làm tròn lên; điều kiện lossless; gộp độ lệch vào tổng activation trong vec_dot.
10. **[Bài 10: Fold Hadamard và runtime FWHT](theory_10_hadamard_folding)**
   * Định lý $W x = W_f \mathbf{H} D x$, ma trận Sylvester, FWHT $O(n \log n)$, `GGML_HINT_SRC0_IS_HADAMARD`, `build_hadamard_input`, embedding latent và tied output, hoán vị value head của Gated DeltaNet, hợp đồng metadata `prism.hadamard.*`.

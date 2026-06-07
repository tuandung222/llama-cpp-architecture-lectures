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

    style T1 fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style T2 fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style T3 fill:#92400e,stroke:#f59e0b,stroke-width:2px,color:#fff
    style T4 fill:#9d174d,stroke:#ec4899,stroke-width:2px,color:#fff
    style T5 fill:#5b21b6,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style T6 fill:#dc2626,stroke:#ef4444,stroke-width:2px,color:#fff
    style T7 fill:#059669,stroke:#10b981,stroke-width:2px,color:#fff
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
   * Information theory log₂(3), absmean quantization, addition-only matmul, scale absorption, STE, LUT mpGEMM.

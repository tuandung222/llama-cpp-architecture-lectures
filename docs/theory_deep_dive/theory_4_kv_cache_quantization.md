---
sidebar_position: 5
sidebar_label: "Bài 4: KV Cache Quantization"
---

# Bài 4: KV Cache Quantization

KV Cache thường chiếm 20-50% tổng bộ nhớ inference. Quantizing KV Cache giảm đáng kể memory footprint mà ảnh hưởng không đáng kể đến chất lượng generation.

---

## 1. KV Cache Memory Footprint

$$\text{KV Size} = 2 \times L \times H_{kv} \times D_h \times C \times S$$

Trong đó:
- $L$ = number of layers (32 cho Llama-3-8B)
- $H_{kv}$ = KV heads (8 cho GQA)
- $D_h$ = head dimension (128)
- $C$ = context length
- $S$ = sizeof(type)

**Ví dụ Llama-3-8B, context 4096:**

$$2 \times 32 \times 8 \times 128 \times 4096 \times 2 = 1,073,741,824 \text{ bytes} = 1.0 \text{ GB (FP16)}$$

---

## 2. Type-0 vs Type-1 Cache

llama.cpp hỗ trợ hai loại KV Cache:

| Type | Tên | Đặc điểm |
|:---|:---|:---|
| Type-0 | Classic | Standard key/value cache cho transformer |
| Type-1 | Recurrent/SSM | Cho recurrent architectures (Mamba, RWKV) |

Type-0 là phổ biến nhất, dùng cho tất cả transformer-based models.

---

## 3. Quantization Options cho KV Cache

```c
params.type_k = GGML_TYPE_F16;   // Default: FP16 keys
params.type_v = GGML_TYPE_F16;   // Default: FP16 values

// Tiết kiệm bộ nhớ:
params.type_k = GGML_TYPE_Q8_0;  // Q8_0 keys: ~50% reduction
params.type_v = GGML_TYPE_Q4_0;  // Q4_0 values: ~75% reduction
```

| Config | Memory (4K ctx) | Quality Impact |
|:---|:---|:---|
| K:F16, V:F16 | 1.0 GB | Baseline |
| K:Q8_0, V:Q8_0 | 0.53 GB | Minimal |
| K:Q8_0, V:Q4_0 | 0.35 GB | Slight |
| K:Q4_0, V:Q4_0 | 0.28 GB | Noticeable |

---

## 4. Attention Quality Under KV Quantization

Quantization error trong KV Cache ảnh hưởng đến attention scores:

$$\hat{A}_{ij} = \text{softmax}\left(\frac{\hat{K}_j^T Q_i}{\sqrt{d}}\right) \neq A_{ij} = \text{softmax}\left(\frac{K_j^T Q_i}{\sqrt{d}}\right)$$

**Thực nghiệm**: Q8_0 KV Cache thường gây < 0.1 perplexity increase, trong khi Q4_0 có thể gây 0.3-0.5 PPL increase.

---

## 5. Sliding Window & Eviction

Với sliding window attention (window size $W$):

$$\text{KV Size} = 2 \times L \times H_{kv} \times D_h \times W \times S$$

Memory cố định bất kể tổng số tokens generated. Tokens ngoài window bị evict.

---

## 💡 Đúc kết

KV Cache quantization là kỹ thuật thiết thực nhất để giảm memory: Q8_0 KV Cache giảm 50% memory mà gần như không ảnh hưởng chất lượng. Sliding window attention giúp memory cố định cho long generation.

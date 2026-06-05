---
sidebar_position: 4
sidebar_label: "Bài 3: Attention Mechanism trên CPU"
---

# Bài 3: Attention Mechanism trên CPU

Self-attention là phép tính nặng nhất trong LLM inference. Bài này phân tích cách llama.cpp tối ưu attention cho CPU, bao gồm KV Cache, Flash Attention adaptation, và RoPE mathematics.

---

## 1. Standard Attention: Độ phức tạp

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Pha | Compute | Memory |
|:---|:---|:---|
| $QK^T$ | $O(n^2 d)$ | $O(n^2)$ cho attention matrix |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Attn $\times V$ | $O(n^2 d)$ | $O(nd)$ |

Với $n = 8192$ (long context) và $d = 128$, attention matrix chiếm $8192^2 \times 4 = 256$ MB. Đây là bottleneck.

---

## 2. KV Cache: Incremental Computation

Trong decode phase (batch size = 1), chỉ cần tính Q cho token mới, K và V được append vào cache:

```
Step t:
  Q_new = W_q @ h_t           (vector, d dimensions)
  K_new = W_k @ h_t           (vector, d dimensions)
  V_new = W_v @ h_t           (vector, d dimensions)
  
  K_cache = [K_cache; K_new]  (append, shape: t × d)
  V_cache = [V_cache; V_new]  (append, shape: t × d)
  
  scores = Q_new @ K_cache^T / sqrt(d_k)   (vector, t scores)
  attn = softmax(scores)                      (vector, t weights)
  output = attn @ V_cache                     (vector, d dimensions)
```

Chi phí mỗi step: $O(t \cdot d)$ thay vì $O(n^2 \cdot d)$.

---

## 3. Flash Attention cho CPU

Flash Attention (Dao et al., 2022) giảm memory từ $O(n^2)$ xuống $O(n)$ bằng tiling:

**Ý tưởng**: Chia K, V thành blocks, tính partial attention cho mỗi block, merge kết quả online.

```
For each K_block, V_block:
  S_block = Q @ K_block^T / sqrt(d_k)
  m_block = max(S_block)           # Local max for numerical stability
  P_block = exp(S_block - m_block)
  
  # Online merge với previous blocks
  m_new = max(m_old, m_block)
  P_new = P_old * exp(m_old - m_new) + P_block * exp(m_block - m_new)
  
Attn = P_new / sum(P_new) @ V_all_blocks
```

Trên CPU, llama.cpp implement Flash Attention qua `ggml_flash_attn_ext()` trong `ggml.c`.

---

## 4. RoPE Mathematics

**Rotary Position Embedding** (RoPE) injects position information bằng rotation:

$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \end{pmatrix}$$

Trong đó $\theta_i = \text{base}^{-2i/d}$, $m$ là position index, base thường là 10000 hoặc 500000 (Llama-3).

**Ưu điểm**: Attention score phụ thuộc vào **relative position** $(m - n)$, không cần additive bias.

---

## 5. GQA và MQA: Memory Savings

| Attention Type | KV Heads | KV Cache Memory (8B, 4K ctx) |
|:---|:---|:---|
| MHA (Multi-Head) | 32 (= Q heads) | 1.0 GB |
| GQA (Grouped-Query) | 8 | 0.25 GB |
| MQA (Multi-Query) | 1 | 0.03 GB |

GQA giảm KV Cache 4x so với MHA, với chất lượng gần như tương đương.

---

## 💡 Đúc kết

Attention trên CPU được tối ưu qua ba kỹ thuật: KV Cache (tránh tính lại), Flash Attention (giảm memory), và GQA (giảm KV heads). RoPE cung cấp relative position encoding mà không cần learned parameters.

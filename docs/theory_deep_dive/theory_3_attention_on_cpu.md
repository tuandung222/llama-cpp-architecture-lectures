---
sidebar_position: 4
sidebar_label: "Bài 3: Attention Mechanism trên CPU"
---

# Bài 3: Attention Mechanism trên CPU

Self-attention là phép tính có độ phức tạp tăng nhanh nhất theo độ dài context trong LLM inference. Bài này phân tích cách llama.cpp tối ưu attention cho CPU, bao gồm KV Cache, Flash Attention (với kernel CPU thật trong `ggml/src/ggml-cpu/ops.cpp`), toán học RoPE và hai biến thể RoPE mà llama.cpp hỗ trợ, cùng GQA/MQA.

---

## 1. Standard Attention: Độ phức tạp

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

với $M$ là mask ($0$ cho cặp hợp lệ, $-\infty$ cho cặp bị che: tương lai, sequence khác, ngoài cửa sổ trượt). Cho một head, $n$ token, head dim $d$:

| Pha | Compute | Memory |
|:---|:---|:---|
| $QK^T$ | $O(n^2 d)$ | $O(n^2)$ cho attention matrix |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| Attn $\times V$ | $O(n^2 d)$ | $O(nd)$ |

Với $n = 8192$ (long context) và $d = 128$, attention matrix F32 của **một head** chiếm $8192^2 \times 4 = 256$ MiB. Nhân với 32 head, con số là 8 GiB cho một layer nếu vật chất hóa toàn bộ. Đây là bottleneck bộ nhớ của prefill dài, và là động lực của Flash Attention.

---

## 2. KV Cache: Incremental Computation

Trong decode phase (mỗi step 1 token), chỉ cần tính Q, K, V cho token mới; K và V được ghi thêm vào cache:

```
Step t:
  Q_new = W_q @ h_t           (vector, d dimensions)
  K_new = W_k @ h_t           (vector, d dimensions)
  V_new = W_v @ h_t           (vector, d dimensions)

  K_cache = [K_cache; K_new]  (shape: t x d)
  V_cache = [V_cache; V_new]  (shape: t x d)

  scores = Q_new @ K_cache^T / sqrt(d_k)     (vector, t scores)
  attn = softmax(scores + mask)               (vector, t weights)
  output = attn @ V_cache                     (vector, d dimensions)
```

Chi phí attention mỗi step: $O(t \cdot d)$ thay vì $O(t^2 \cdot d)$ nếu chạy lại attention cho cả $t$ token. Trong llama.cpp, "append" không phải là nối mảng: K, V mới được ghi vào các ô do `llama_kv_cache::find_slot()` chọn, bằng op `ggml_set_rows()` với tensor chỉ số `k_idxs`/`v_idxs`; các ô không cần liền kề và mask lo phần còn lại (Bài 4 chính, Phần 4.4).

Ở decode, attention của một query là một **matrix-vector** (GEMV) trên K cache, và phải đọc toàn bộ $t \times d$ phần tử K cùng $t \times d$ phần tử V của mỗi head mỗi layer. Khi context dài, lượng byte này có thể vượt cả lượng byte trọng số của layer: KV Cache trở thành nút thắt băng thông thứ hai sau trọng số.

---

## 3. Flash Attention cho CPU

Flash Attention (Dao et al., 2022) giảm memory từ $O(n^2)$ xuống $O(n)$ bằng cách không bao giờ vật chất hóa ma trận attention, dựa trên **online softmax**.

**Toán học của online softmax.** Với một query, duyệt các key $j = 1, 2, \dots$ và duy trì ba đại lượng: $M$ (max hiện tại của các score), $S$ (tổng $\sum e^{s_j - M}$), và vector $O$ (tổng $\sum e^{s_j - M} v_j$). Khi gặp score mới $s$:

$$M' = \max(M, s), \quad m_s = e^{M - M'}, \quad v_s = e^{s - M'}$$
$$S' = S \cdot m_s + v_s, \qquad O' = O \cdot m_s + v_s \cdot v$$

Cuối cùng $\text{output} = O / S$. Bản chất của công thức nằm ở việc: khi max tăng, mọi đóng góp cũ được nhân lại với $e^{M - M'}$ để quy về cùng "gốc" mới, nên kết quả chính xác tuyệt đối (không phải xấp xỉ) và ổn định số học. Chú ý cả $S$ **và** $O$ đều phải được scale lại; chỉ scale tổng xác suất là sai.

**Hiện thực trong llama.cpp.** Op được định nghĩa trong `ggml/src/ggml.c` bởi `ggml_flash_attn_ext()` (chỉ tạo nút đồ thị); kernel CPU là `ggml_compute_forward_flash_attn_ext()` trong `ggml/src/ggml-cpu/ops.cpp`, với lõi `ggml_compute_forward_flash_attn_ext_f16_one_chunk()`. Pseudocode rút gọn cho mỗi hàng query (các thread chia nhau các hàng query và head):

```text
S = 0; M = -inf; VKQ32[0..DV) = 0            # accumulator trong work buffer của thread
q_to_vec_dot(q_row, Q_q)                     # chuyển Q sang vec_dot_type của kiểu K (ví dụ F16, Q8_0)
for each KV cell ic in [0, n_kv):
    mv = mask[ic] (+ ALiBi slope)             # -inf -> bỏ qua ô này
    if mv == -inf: continue
    kq_vec_dot(DK, &s, k_row(ic), Q_q)        # dot product, dùng kernel của kiểu K (K cache có thể quantized)
    s = s * scale (+ softcap) + mv
    Mold = M
    if s > M:  M = s; ms = expf(Mold - M); ggml_vec_scale_f32(DV, VKQ32, ms)   # scale lại O
    else:      vs = expf(s - M)
    ggml_vec_mad_f32(DV, VKQ32, v_row(ic) (đã to_float nếu V quantized), vs)   # O += vs * v
    S = S*ms + vs
VKQ32 *= 1/S                                  # chuẩn hóa cuối
```

Ba quan sát từ mã thật:

- Kernel CPU đi theo **từng ô KV một** cho mỗi query (streaming), không chia tile lớn như kernel GPU; điều đó vẫn đủ để bộ nhớ chỉ là $O(d)$ mỗi thread.
- Score $QK$ được tính bằng **dot product của kiểu K cache** (`kq_vec_dot`), nên K cache quantized (Q8_0, Q4_0...) được dùng trực tiếp mà không cần dequantize cả cache. V cache quantized được chuyển sang float từng hàng (`v_to_float`).
- Đây là lý do `llama-context.cpp` báo lỗi "quantized V cache was requested, but this requires Flash Attention" khi tắt flash attention: đường không-FA dùng `ggml_mul_mat` với V **chuyển vị**, không hỗ trợ V quantized.

Trong graph builder, `llm_graph_context::build_attn_mha()` (`src/llama-graph.cpp`) chọn `ggml_flash_attn_ext()` khi `cparams.flash_attn` bật, ngược lại dùng chuỗi `ggml_mul_mat` (KQ), `ggml_soft_max_ext` (mask, scale), `ggml_mul_mat` (KQV). Mặc định `flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO` (tự bật khi backend hỗ trợ); CLI dùng `-fa on|off|auto`.

---

## 4. RoPE Mathematics

**Rotary Position Embedding** (RoPE) mã hóa vị trí bằng phép quay từng cặp thành phần:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \qquad \theta_i = \text{base}^{-2i/d}, \; i = 0, \dots, \tfrac{d}{2} - 1$$

Trong đó $m$ là position index, $d$ là số chiều được quay (`n_rot`, thường bằng head dim), base thường là 10000 (Llama 2) hoặc 500000 (Llama 3, key `llama.rope.freq_base`). Kernel CPU tính $\theta_i$ theo cấp số nhân với công bội `theta_scale = powf(freq_base, -2.0f/n_dims)` (`ggml/src/ggml-cpu/ops.cpp`), và hàm `rope_yarn()` xử lý các biến thể mở rộng context (YaRN, `freq_scale`).

**Ưu điểm**: Vì tích của hai phép quay là phép quay theo hiệu góc, $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$: attention score chỉ phụ thuộc vào **relative position** $(n - m)$, không cần additive bias hay learned parameters.

**Hai cách ghép cặp.** llama.cpp có hai chế độ chính (`enum llama_rope_type` trong `include/llama.h`):

| Chế độ | Cặp được quay | Dùng bởi (theo `llama_model_rope_type()` trong `src/llama-model.cpp`) |
|:---|:---|:---|
| `LLAMA_ROPE_TYPE_NORM` (0) | $(x_{2i}, x_{2i+1})$, hai phần tử kề nhau | `LLM_ARCH_LLAMA` và nhiều kiến trúc kiểu Llama |
| `LLAMA_ROPE_TYPE_NEOX` | $(x_i, x_{i + d/2})$, nửa đầu với nửa sau | GPT-NeoX, Qwen2, Qwen3, ... |

Còn có `MROPE`, `IMROPE`, `VISION` cho mô hình đa phương thức. Một chi tiết tinh tế: mã HuggingFace của Llama dùng kiểu "rotate_half" (tức ghép cặp kiểu NEOX), nhưng llama.cpp chạy Llama với `NORM`. Hai cách cho kết quả giống nhau vì bộ chuyển đổi **hoán vị các hàng** của `W_q` và `W_k` khi ghi GGUF (hàm `permute()` trong `conversion/llama.py`), biến cặp $(i, i + d/2)$ thành cặp kề nhau. Port một kiến trúc mới mà chọn sai chế độ RoPE hoặc quên hoán vị là một nguồn lỗi kinh điển: mô hình vẫn chạy nhưng sinh văn bản vô nghĩa.

---

## 5. GQA và MQA: Memory Savings

Với Grouped-Query Attention, $H_q$ query head chia sẻ $H_{kv}$ cặp K/V head, mỗi nhóm $g = H_q / H_{kv}$ query head (hàm `n_gqa()` trong `llama_hparams`). KV Cache tỷ lệ với $H_{kv}$. Cho cấu hình kiểu Llama-3-8B (32 layer, head dim 128, F16, context 4096 token):

| Attention Type | KV Heads | KV Cache Memory (4K ctx, F16) |
|:---|:---|:---|
| MHA (Multi-Head) | 32 (= Q heads) | 2.0 GiB |
| GQA (Grouped-Query, Llama-3-8B thật) | 8 | 0.5 GiB |
| MQA (Multi-Query) | 1 | 64 MiB |

Tính kiểm tra cho GQA: $2 \times 32 \times 8 \times 128 \times 4096 \times 2 = 536{,}870{,}912$ byte $= 0.5$ GiB. GQA giảm KV Cache 4 lần so với MHA, với chất lượng gần như tương đương. Ngoài bộ nhớ, GQA còn giảm **băng thông** đọc K/V khi decode cùng tỷ lệ, và ở FA kernel mỗi hàng K/V được dùng lại cho $g$ query head.

Một bước xa hơn là **MLA** (Multi-head Latent Attention, DeepSeek-V2/V3): cache một vector latent nén thay vì K/V đầy đủ. llama.cpp hỗ trợ MLA qua tham số `v_mla` của `build_attn_mha()`.

---

## 💡 Đúc kết

Attention trên CPU được tối ưu qua các kỹ thuật: KV Cache (tránh tính lại, ghi bằng `ggml_set_rows` vào các ô bất kỳ), Flash Attention (online softmax, kernel `ggml_compute_forward_flash_attn_ext_f16_one_chunk` đọc trực tiếp K cache quantized), và GQA/MQA/MLA (giảm số K/V cần lưu và đọc). RoPE cung cấp relative position encoding không cần learned parameters, nhưng đòi hỏi chọn đúng chế độ ghép cặp (`NORM` hay `NEOX`) khi port mô hình.

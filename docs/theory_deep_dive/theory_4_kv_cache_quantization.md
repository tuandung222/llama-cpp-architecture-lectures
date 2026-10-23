---
sidebar_position: 5
sidebar_label: "Bài 4: KV Cache Quantization"
---

# Bài 4: KV Cache Quantization

Khi context dài, KV Cache có thể chiếm phần lớn bộ nhớ inference, thậm chí vượt kích thước trọng số đã quantize. Quantizing KV Cache giảm đáng kể memory footprint (và băng thông đọc khi decode), đổi lại một sai số trên attention score cần được hiểu và kiểm soát. Bài này phân tích công thức bộ nhớ, các kiểu cache mà llama.cpp cho phép, toán học của sai số, và kỹ thuật **xoay Hadamard** mà llama.cpp áp dụng tự động cho cache quantized.

---

## 1. KV Cache Memory Footprint

$$\text{KV Size} = 2 \times L \times H_{kv} \times D_h \times C \times S$$

Trong đó:
- $L$ = number of layers (32 cho Llama-3-8B)
- $H_{kv}$ = KV heads (8 cho GQA)
- $D_h$ = head dimension (128)
- $C$ = số ô cache (context length)
- $S$ = số byte trung bình mỗi phần tử: 2 cho F16; với kiểu block là bpw/8, tức 1.0625 cho Q8_0, 0.5625 cho Q4_0

**Ví dụ Llama-3-8B, context 4096:**

$$2 \times 32 \times 8 \times 128 \times 4096 \times 2 = 536{,}870{,}912 \text{ bytes} = 0.5 \text{ GiB (F16)}$$

Tức 128 KiB cho mỗi token. Ở context 32K là 4 GiB, ở 128K (context tối đa của Llama-3.1-8B) là 16 GiB, gấp hơn 3 lần kích thước trọng số Q4_K_M (4.58 GiB). Khi tạo context, `llama_kv_cache` in đúng con số này trong log (`size = ... MiB (... cells, ... layers, ...), K (f16): ... MiB, V (f16): ... MiB`).

---

## 2. Không chỉ có KV Cache: các loại memory trong llama.cpp

Thay cho cách phân loại cũ theo "kiểu cache", llama.cpp hiện có interface `llama_memory_i` (`src/llama-memory.h`) với nhiều cài đặt, do `llama_model::create_memory()` chọn theo kiến trúc:

| Lớp | Trạng thái lưu | Bộ nhớ theo context |
|:---|:---|:---|
| `llama_kv_cache` | K, V của từng token, từng layer attention | Tuyến tính theo $C$ |
| `llama_kv_cache_iswa` | Hai `llama_kv_cache`: `kv_base` (layer full attention) và `kv_swa` (layer sliding window) | Tuyến tính theo $C$ cho phần base, gần hằng số cho phần SWA |
| `llama_memory_recurrent` | State cố định mỗi sequence (`r_l`, `s_l` cho mỗi layer: conv state và SSM/WKV state) | **Hằng số** theo $C$, tuyến tính theo số sequence |
| `llama_memory_hybrid` (+ `_iswa`, `_idx`) | Kết hợp một KV cache cho layer attention và một recurrent memory cho layer SSM | Tuyến tính theo $C$ nhưng chỉ cho số ít layer attention |

Các mô hình hybrid (Jamba, Falcon-H1, Nemotron-H, Qwen3-Next, Granite hybrid...) là một chiến lược "nén KV" ở mức kiến trúc: chỉ một phần nhỏ layer là attention, nên KV Cache nhỏ đi tương ứng mà không cần quantize.

---

## 3. Quantization Options cho KV Cache

Kiểu K và V được đặt độc lập qua `llama_context_params::type_k`, `type_v` (mặc định `GGML_TYPE_F16`, cả hai đánh dấu `[EXPERIMENTAL]` trong `include/llama.h`), hoặc qua CLI `-ctk` / `--cache-type-k` và `-ctv` / `--cache-type-v`. Danh sách giá trị hợp lệ lấy từ mảng `kv_cache_types` trong `common/arg.cpp`: **f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1**.

```c
cparams.type_k = GGML_TYPE_Q8_0;   // K: ~53% bộ nhớ của F16 (8.5/16)
cparams.type_v = GGML_TYPE_Q4_0;   // V: ~28% bộ nhớ của F16 (4.5/16)
cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // bắt buộc khi V quantized
```

Ba ràng buộc thực tế:

1. **V quantized cần Flash Attention.** Không có FA, V được lưu chuyển vị (tham số `v_trans = !cparams.flash_attn` khi tạo cache) để dùng `ggml_mul_mat`, và đường đó không hỗ trợ V quantized; `llama-context.cpp` báo lỗi "quantized V cache was requested, but this requires Flash Attention".
2. **Head dim phải chia hết cho block size** (32 với các kiểu trên), vì mỗi hàng K của một head được lượng tử thành các block nguyên.
3. **Context shift với K quantized** tốn hơn: khi dịch vị trí (ví dụ `--context-shift`), RoPE phải được áp lại lên K đã lưu. Hàm `build_rope_shift()` trong `llama-kv-cache.cpp` với K quantized phải `ggml_cast` sang F32, quay lại, rồi `ggml_cpy` lượng tử lại, mỗi lần như vậy cộng thêm sai số.

Bộ nhớ cho Llama-3-8B, context 4096 (K và V mỗi bên 0.25 GiB ở F16):

| Config | Memory (4K ctx) | Tỷ lệ so với F16 |
|:---|:---|:---|
| K:F16, V:F16 | 0.50 GiB | 100% |
| K:Q8_0, V:Q8_0 | 0.27 GiB | 53% |
| K:Q8_0, V:Q4_0 | 0.20 GiB | 41% |
| K:Q4_0, V:Q4_0 | 0.14 GiB | 28% |

---

## 4. Attention Quality Under KV Quantization

Quantization error trong KV Cache ảnh hưởng đến attention scores:

$$\hat{A}_{ij} = \text{softmax}_j\left(\frac{Q_i^{\top} \hat{K}_j}{\sqrt{d}}\right) \neq A_{ij} = \text{softmax}_j\left(\frac{Q_i^{\top} K_j}{\sqrt{d}}\right)$$

**Sai số trên score.** Viết $\hat{K}_j = K_j + \epsilon_j$ với các thành phần sai số độc lập, trung bình 0, variance $\Delta^2/12$ (Theory 1). Sai số của score là

$$\delta s_{ij} = \frac{Q_i^{\top} \epsilon_j}{\sqrt{d}}, \qquad \text{Var}(\delta s_{ij}) = \frac{\|Q_i\|_2^2}{d} \cdot \frac{\Delta^2}{12}$$

Đọc công thức này theo nghĩa thực tế: sai số score tỷ lệ với **độ lớn của query** và với bước lượng tử $\Delta$ của K. Vì softmax là hàm mũ, một sai số cộng $\delta s$ trên logit nhân xác suất lên $e^{\delta s}$; nên sai số của K bị **khuếch đại** qua softmax.

**Sai số trên V** thì khác: output là $\sum_j A_{ij} \hat{V}_j$, một trung bình có trọng số. Sai số độc lập của các $V_j$ bị **trung bình hóa**: variance của sai số output giảm theo $\sum_j A_{ij}^2 \leq 1$. Đây là lý do thực tiễn phổ biến là giữ K ở độ chính xác cao hơn V (ví dụ K:Q8_0, V:Q4_0), chứ không phải ngược lại.

**Vấn đề outlier của K.** Các vector key của LLM thường có vài kênh (channel) có độ lớn vượt trội, nhất là sau RoPE. Với block quantization dùng scale theo max (Q8_0, Q4_0), một outlier trong block làm $\Delta$ lớn cho cả 32 phần tử còn lại. Phần 5 trình bày cách llama.cpp xử lý.

**Đo chất lượng.** Các con số "tăng bao nhiêu PPL" phụ thuộc mô hình, độ dài context và dữ liệu, nên cần tự đo: chạy `llama-perplexity` với `-ctk`/`-ctv` khác nhau và so KL divergence với cấu hình F16 (Theory 5). Về định tính, Q8_0 cho K và V thường gần như không phân biệt được với F16, trong khi Q4_0 cho K là cấu hình rủi ro nhất.

---

## 5. Xoay Hadamard: giảm outlier trước khi lượng tử

Mã nguồn hiện tại của llama.cpp có một kỹ thuật ít được biết đến nhưng rất hiệu quả, theo tinh thần của QuaRot/QuIP: khi K (hoặc V) cache là kiểu quantized và head dim chia hết cho 64, `llama_kv_cache` tự tạo ma trận **Hadamard chuẩn hóa** $H$ (hàm `ggml_gen_hadamard()` trong `src/llama-kv-cache.cpp`, dựng theo công thức đệ quy Sylvester, phần tử $\pm 1/\sqrt{n}$) và áp vào trước khi ghi cache.

Ma trận $H$ trực giao và đối xứng: $H^{\top} H = I$, $H = H^{\top}$, nên $H^{-1} = H$. Do đó, nếu quay **cả** Q và K bằng cùng $H$:

$$(H q)^{\top} (H k) = q^{\top} H^{\top} H k = q^{\top} k$$

attention score không đổi về mặt toán học, nhưng vector $Hk$ được lượng tử thay cho $k$. Phép biến đổi Hadamard "trải" năng lượng của một outlier ra tất cả $n$ thành phần (mỗi thành phần của $Hk$ là tổng có dấu của mọi thành phần $k$, chia $\sqrt{n}$), nên phân phối sau khi quay gần Gaussian và max của mỗi block nhỏ đi đáng kể so với độ lệch chuẩn: $\Delta$ giảm, sai số giảm. Với V, giá trị được quay trước khi ghi cache, và output attention được quay ngược lại bằng chính $H$, vì $\sum_j A_{ij} H v_j = H \sum_j A_{ij} v_j$.

Trong đồ thị (`src/llama-graph.cpp`), điều này hiện ra dưới dạng:

```cpp
if (inp->self_k_rot) {
    q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);   // H q
    k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);   // H k, rồi mới ghi vào cache quantized
}
if (inp->self_v_rot) {
    v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);   // H v
}
// ... attention ...
if (inp->self_v_rot) {
    cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);       // H (sum A H v) = sum A v
}
```

Kích thước khối quay: với K, bắt đầu từ 64 và nhân đôi chừng nào head dim còn chia hết (ví dụ head dim 128 dùng $H_{128}$); với V cố định 64 (comment trong mã ghi nhận thực nghiệm rằng ma trận nhỏ hơn cho V tốt hơn). Có thể tắt toàn cục bằng biến môi trường `LLAMA_ATTN_ROT_DISABLE=1` hoặc cho từng mô hình qua `llama_model_set_attn_rot()`. Khi cache là F16/BF16/F32, không có phép quay nào được thêm vào.

---

## 6. Sliding Window & Eviction

Với sliding window attention (window size $W$), layer SWA chỉ cần giữ khoảng $W$ token gần nhất:

$$\text{KV Size}_{\text{SWA layers}} = 2 \times L_{\text{swa}} \times H_{kv} \times D_h \times C_{\text{swa}} \times S$$

Trong `llama_kv_cache_iswa`, số ô của cache SWA là

$$C_{\text{swa}} = \text{PAD}_{256}\big(\min(C,\; W \cdot n_{\text{seq}} + n_{\text{ubatch}})\big)$$

($n_{\text{seq}} = n_{\text{seq\_max}}$ khi cache unified, ngược lại 1). Số hạng $n_{\text{ubatch}}$ cần thiết vì trong một ubatch, token đầu tiên vẫn phải thấy $W$ token trước nó trong khi các token sau đã được ghi. Phần SWA có bộ nhớ gần như cố định bất kể tổng số tokens generated; tokens ngoài window bị che bởi mask (`llama_hparams::is_masked_swa()`) và ô của chúng được tái sử dụng. Đổi lại, cache SWA rút gọn không cho phép quay lui về một vị trí cũ hơn cửa sổ (ví dụ để tái sử dụng prompt cache), nên tồn tại tùy chọn `--swa-full` dùng cache SWA kích thước đầy đủ.

Với layer full attention, llama.cpp không tự "evict" token cũ; khi cache đầy, ứng dụng phải chọn: báo lỗi (llama_decode trả về 1), xóa bớt bằng `llama_memory_seq_rm()`, hoặc dịch vị trí bằng `llama_memory_seq_add()` (context shift, cần RoPE lại K như ở Phần 3).

---

## 💡 Đúc kết

KV Cache quantization là kỹ thuật thiết thực nhất để giảm memory ở context dài: Q8_0 giảm gần một nửa bộ nhớ mà gần như không ảnh hưởng chất lượng, và kết hợp K:Q8_0, V:Q4_0 giảm còn khoảng 41%. Toán học cho thấy K nhạy hơn V (sai số K bị softmax khuếch đại, sai số V bị trung bình hóa), và outlier là kẻ thù chính; llama.cpp tự động xoay Hadamard Q, K, V khi cache quantized để giảm outlier mà không đổi kết quả attention. Ở mức kiến trúc, sliding window và mô hình hybrid giảm KV Cache mà không cần quantize.

---
sidebar_position: 8
sidebar_label: "Bài 7: Toán học Ternary Quantization & BitNet b1.58"
---

# Bài 7: Toán học Ternary Quantization & BitNet b1.58

Bài này đi sâu vào toán học đằng sau inference ternary: giới hạn thông tin của weight 3 trạng thái, cách các định dạng thực tế tiến gần giới hạn đó, phép nhân ma trận chỉ dùng phép cộng, cách xử lý scale factor, và kỹ thuật Lookup Table (LUT) cho mixed-precision GEMM.

---

## Phần 1. Giới hạn Thông tin của Ternary Weights

### 1.1. Information-Theoretic Analysis

Mỗi weight ternary nhận một trong 3 giá trị $\{-1, 0, +1\}$. Nếu ba giá trị đồng xác suất, entropy Shannon cho số bit tối thiểu để mã hóa là:

$$H = \log_2(3) \approx 1.585 \text{ bits/weight}$$

Đây là **giới hạn lý thuyết** cho mã hóa độ dài cố định. (Nếu phân phối không đều, ví dụ 31% số 0 như Phần 2.3, entropy thực còn thấp hơn một chút, nhưng muốn khai thác điều đó cần mã hóa entropy có độ dài thay đổi, không thân thiện với SIMD.) Mọi cách đóng gói thực tế đều tốn nhiều hơn:

| Phương pháp | Bits/weight | Overhead so với 1.585 | Cấu trúc |
|:---|:---|:---|:---|
| Giới hạn Shannon | 1.585 | 0% | - |
| TL2 (bitnet.cpp) | 1.667 | +5.2% | 3 weights = 1 bit dấu + 4-bit index |
| TQ1_0 (llama.cpp) | 1.6875 | +6.5% | 54 byte cho 256 weights |
| I2_S (bitnet.cpp) | 2.000 | +26.2% | 2 bit/weight, một scale FP32 cho cả tensor |
| TL1 (bitnet.cpp) | 2.000 | +26.2% | 2 weights = 4-bit index |
| TQ2_0 (llama.cpp) | 2.0625 | +30.1% | 66 byte cho 256 weights |

Kiểm tra các con số của llama.cpp từ `ggml/src/ggml-common.h`:

- `block_tq1_0` gồm `qs[48]` (mỗi byte chứa 5 trits, vì $3^5 = 243 \le 256$), `qh[4]` (mỗi byte chứa 4 trits) và một scale `ggml_half d`. Tổng $48 + 4 + 2 = 54$ byte cho $48 \times 5 + 4 \times 4 = 256$ weights, tức $54 \times 8 / 256 = 1.6875$ bpw.
- `block_tq2_0` gồm `qs[64]` (2 bit/trit) và `ggml_half d`. Tổng 66 byte cho 256 weights, tức $66 \times 8 / 256 = 2.0625$ bpw.

Một chi tiết tinh tế của TQ1_0: 5 trits $(t_0, \ldots, t_4) \in \{0,1,2\}^5$ được ghép thành số $q = \sum_k t_k 3^{4-k} \in [0, 242]$ rồi lưu dưới dạng **phân số cố định** $\lceil 256 q / 243 \rceil$. Khi giải mã, nhân byte với 3 (phép nhân uint8) và lấy byte cao sẽ "đẩy" từng trit ra lần lượt, thay vì phải chia lấy dư cho 3. Xem `quantize_row_tq1_0_ref()` và `ggml_vec_dot_tq1_0_q8_K`.

TL2 đạt gần giới hạn nhất (overhead 5.2%) bằng cách nén **3 weights vào 5 bit**: $3^3 = 27 \le 32$, nhưng thay vì đánh chỉ số 27 giá trị, TL2 tận dụng tính đối xứng để dùng 1 bit dấu và 4 bit index vào bảng 14 entry (Phần 6.3).

### 1.2. Tại sao 1.58-bit tốt hơn 4-bit?

So sánh memory cho phần weight của một model $7 \times 10^9$ tham số (bỏ qua embedding):

$$M_{\text{FP16}} = 7 \times 10^9 \times 2 \text{ byte} = 14 \text{ GB}$$

$$M_{\text{Q4\_K\_M}} \approx 7 \times 10^9 \times \frac{4.85}{8} \approx 4.24 \text{ GB}$$

$$M_{\text{TL2}} = 7 \times 10^9 \times \frac{1.67}{8} \approx 1.46 \text{ GB}$$

Ở đây 4.85 bpw là mức trung bình điển hình của Q4_K_M (một hỗn hợp Q4_K và Q6_K). Ternary tiết kiệm khoảng **12.5 GB** so với FP16 và khoảng **2.8 GB** so với Q4_K_M. Vì pha decode bị giới hạn bởi bandwidth, lượng byte đọc mỗi token giảm khoảng $4.85 / 1.67 \approx 2.9$ lần so với Q4_K_M, đó là trần lý thuyết của speedup decode khi chỉ xét băng thông.

---

## Phần 2. Absmean Quantization Mathematics

### 2.1. Công thức

Cho ma trận weight $W$ với $N$ phần tử, absmean quantization:

$$s = \frac{1}{N} \sum_{i=1}^{N} \lvert W_i \rvert = \frac{\lVert W \rVert_1}{N}$$

$$\hat{W}_i = \text{clamp}\!\left(\text{round}\!\left(\frac{W_i}{s + \epsilon}\right), -1, 1\right)$$

Trong BitNet b1.58, $s$ được tính trên **toàn bộ ma trận** (per-tensor), không phải per-row hay per-block.

### 2.2. So sánh với Symmetric Quantization (Max-Abs)

| Phương pháp | Scale factor | Miền của $W_i / s$ trước clamp |
|:---|:---|:---|
| Symmetric (max-abs) | $s = \max_i \lvert W_i \rvert$ | $[-1, 1]$ |
| Absmean | $s = \text{mean}_i \lvert W_i \rvert$ | $[-N, N]$ trong trường hợp xấu nhất |

Cận $N$ đạt được khi toàn bộ khối lượng $\lVert W \rVert_1$ dồn vào một phần tử; với phân phối thực tế, $\max \lvert W_i \rvert / s$ chỉ cỡ vài đơn vị. Absmean chủ động **clamp** các weight lớn hơn $1.5 s$ về $\pm 1$. Với max-abs, gần như mọi weight sẽ bị làm tròn về 0 khi chỉ có 3 mức, nên absmean là lựa chọn hợp lý cho ternary. Việc clamp các outlier là chấp nhận được **vì model được train với chính ràng buộc này**.

### 2.3. Threshold Behavior

Với weight có phân phối Gaussian $W \sim \mathcal{N}(0, \sigma^2)$, kỳ vọng trị tuyệt đối là:

$$s = \mathbb{E}\lvert W \rvert = \sigma\sqrt{2/\pi} \approx 0.798\,\sigma$$

Weight bị làm tròn về 0 khi $\lvert W_i / s \rvert < 1/2$, tức $\lvert W_i \rvert < s/2 \approx 0.399\,\sigma$. Xác suất tương ứng:

$$P(\lvert W \rvert < 0.399\,\sigma) = 2\Phi(0.399) - 1 = \text{erf}\!\left(\frac{0.399}{\sqrt{2}}\right) \approx 0.31$$

Vậy khoảng **31%** weight sẽ bằng 0 (không phải 38% như đôi khi được nêu). Về nguyên lý, đó là 31% phép toán có thể bỏ qua; trên thực tế các kernel SIMD (I2_S, TQ2_0) vẫn xử lý dày đặc cả số 0 vì rẽ nhánh theo từng phần tử phá vỡ vectorization. Phân phối weight của một model BitNet đã train không nhất thiết Gaussian, nên con số này chỉ là ước lượng ban đầu.

---

## Phần 3. Addition-Only Matrix Multiplication

### 3.1. Proof

Cho $W \in \{-1, 0, +1\}^{m \times k}$ và $x \in \mathbb{R}^k$:

$$y_i = \sum_{j=1}^{k} W_{ij} \cdot x_j = \sum_{j: W_{ij}=+1} x_j - \sum_{j: W_{ij}=-1} x_j$$

Không có phép nhân nào. Mỗi output element cần tối đa $k - 1$ phép cộng/trừ.

### 3.2. Complexity Analysis

| Phương pháp | Multiplications mỗi output | Additions mỗi output | Ghi chú |
|:---|:---|:---|:---|
| FP16 GEMV | $k$ (float) | $k-1$ | $2k - 1$ FLOPs |
| Q4_K trên CPU llama.cpp | $k$ (int8 MAD) + $O(k/32)$ (nhân nguyên với scale 6-bit của sub-block) + $O(k/256)$ (float, scale của super-block) | $\approx k$ | `ggml_vec_dot_q4_K_q8_K` |
| BitNet (addition-only) | 0 (+ 1 nhân scale cuối) | $\le k - 1$ | Trên lý thuyết |

### 3.3. SIMD Implementation

Pseudocode khái niệm cho ternary GEMV với mask:

```
// Pseudocode: ternary GEMV dùng mask
for each 32-element block:
    w = load_weights()           // {-1, 0, +1} packed
    positive_mask = (w == +1)
    negative_mask = (w == -1)
    acc += masked_add(activations, positive_mask)
    acc -= masked_add(activations, negative_mask)
    // weight = 0: không đóng góp
```

Các kernel thực tế lại chọn một mẹo đại số hiệu quả hơn: lưu trit dưới dạng $t = w + 1 \in \{0, 1, 2\}$ rồi dùng lệnh nhân-cộng số nguyên có sẵn:

$$\sum_j w_j x_j = \sum_j t_j x_j - \sum_j x_j$$

Trên AVX2, `_mm256_maddubs_epi16(t, x)` tính $\sum t_j x_j$ (unsigned × signed int8) trong một lệnh, còn $\sum_j x_j$ đã được tính sẵn khi lượng tử activation (trường `bsums` của `block_q8_K`). Đây đúng là cách `ggml_vec_dot_tq2_0_q8_K` trong `ggml/src/ggml-cpu/arch/x86/quants.c` hoạt động, và kernel I2_S của bitnet.cpp dùng cùng mã hóa 0/1/2. Trên ARM NEON, vai trò tương ứng do `vdotq_s32` (khi có `__ARM_FEATURE_DOTPROD`) hoặc `vmlal_s8` đảm nhận.

---

## Phần 4. Xử lý Scale Factor

### 4.1. Derivation

BitLinear forward pass cho token $t$:

$$y_t = \hat{W} \hat{x}_t \quad \text{(integer matmul)}$$

$$y^{\text{fp}}_t = y_t \cdot c_t, \qquad c_t = \frac{s_w \cdot \eta_t}{Q_b}$$

với $s_w$ là absmean scale của weight (hằng số theo tensor), $\eta_t = \max_i \lvert x_{t,i} \rvert$ là absmax scale của activation (hằng số theo token), và $Q_b = 2^{b-1} = 128$ cho 8 bit (nhiều cài đặt dùng 127).

### 4.2. Khi nào scale bị triệt tiêu, khi nào không

**Trường hợp 1: output đi thẳng vào RMSNorm.** Với $c > 0$:

$$\text{RMSNorm}(c\,y) = \frac{c\,y}{\sqrt{\frac{1}{d}\sum_i (c\,y_i)^2 + \epsilon}} \odot \gamma \;\approx\; \frac{y}{\text{RMS}(y)} \odot \gamma$$

Hằng số $c$ triệt tiêu (chính xác khi $\epsilon \to 0$) vì RMSNorm bất biến theo phép co giãn dương. Khi đó **không cần** nhân $c$, và cũng **không cần** gộp gì vào $\gamma$: đây là kết luận đúng của lập luận scale-invariance. Lưu ý rằng công thức "$\gamma_{\text{new}} = \gamma \cdot s_w$" mà đôi khi được nêu là mâu thuẫn với chính lập luận này: nếu $c$ đã bị triệt tiêu thì nhân $s_w$ vào $\gamma$ sẽ làm sai kết quả.

**Trường hợp 2: output đi vào residual stream** ($W_O$ của attention, $W_{\text{down}}$ của FFN). Phép cộng residual không bất biến theo scale, nên $c_t$ **phải** được áp dụng. Chi phí chỉ là $m$ phép nhân cho một vector output kích thước $m$, so với $mk$ phép cộng của GEMV: tỷ lệ $1/k$, không đáng kể.

**Trường hợp 3: Q và K.** Điểm attention $q^\top k$ nhân với $c_q c_k$; phần $s_{w_Q} s_{w_K}$ là hằng số và có thể gộp vào hệ số $1/\sqrt{d_h}$ của softmax, còn phần $\eta$ phụ thuộc token vẫn phải nhân.

**Trường hợp 4: FFN có cổng.** Với SwiGLU, $\text{SiLU}(c_g g) \neq c_g\,\text{SiLU}(g)$ nên $c_g$ phải được nhân trước phi tuyến. Với squared ReLU (BitNet b1.58 2B4T), $\text{ReLU}(c\,g)^2 = c^2\,\text{ReLU}(g)^2$ là hàm thuần nhất bậc 2, nên các hằng số dương đi qua phi tuyến và có thể bị `ffn_sub_norm` phía sau triệt tiêu.

Trong upstream llama.cpp, kiến trúc BitNet chọn cách đơn giản và luôn đúng: mỗi linear có một tensor scale 1 phần tử tùy chọn (`wq_s`, `wo_s`, `ffn_down_s`...) và `build_lora_mm()` trong `src/llama-graph.cpp` nhân nó vào bằng `ggml_mul` ngay sau phép nhân ma trận.

### 4.3. So sánh với GGML block quantization

Với Q4_0, mỗi block 32 weight có một scale FP16 riêng. Trên CPU, llama.cpp **không** dequantize ra mảng float: `ggml_vec_dot_q4_0_q8_0` tính tích số nguyên giữa block weight 4-bit và block activation `Q8_0`, rồi nhân với $d_w \cdot d_x$ của từng block:

```c
// Pseudocode: ggml_vec_dot_q4_0_q8_0
sumf = 0
for each block b of 32 values:
    isum = sum_j (nibble(x[b].qs, j) - 8) * y[b].qs[j]   // integer
    sumf += FP16_TO_FP32(x[b].d) * FP16_TO_FP32(y[b].d) * isum
```

Hàm `dequantize_row_q4_0` có tồn tại, nhưng chủ yếu phục vụ chuyển đổi kiểu, `get_rows` và một số đường tính trên GPU. Như vậy khác biệt thật giữa GGML block quant và BitNet là **số lượng scale**: một cặp scale float cho mỗi 32 phần tử, so với một scale cho cả tensor và một scale cho mỗi token.

---

## Phần 5. Straight-Through Estimator (STE)

### 5.1. Forward Pass

$$W_q = Q(W) = s \cdot \text{clamp}\!\left(\text{round}\!\left(\frac{W}{s}\right), -1, 1\right)$$

Hàm $Q$ là hàm bậc thang: đạo hàm bằng 0 hầu khắp nơi và không xác định tại các điểm nhảy, nên gradient thật không dùng được để học.

### 5.2. Backward Pass: STE Approximation

Dạng STE thuần (được dùng trong các bản cài đặt BitNet công khai, viết dưới dạng `w + (Q(w) - w).detach()`):

$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q}$$

Biến thể "clipped STE" chặn gradient ngoài miền lượng tử:

$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q} \cdot \mathbf{1}_{\lvert W/s \rvert \le 1}$$

Cả hai đều là **ước lượng**, không cố gắng tính đúng gradient của hàm bậc thang.

### 5.3. Hệ quả của clipped STE

| Region | Gradient | Effect |
|:---|:---|:---|
| $\lvert W/s \rvert \le 1$ | Đi qua | Weight được optimize bình thường |
| $\lvert W/s \rvert > 1$ | Bằng 0 | Weight bị "đóng băng" tạm thời |

Việc đóng băng không vĩnh viễn: $s$ là trung bình trên toàn tensor nên thay đổi khi các weight khác được cập nhật, và weight decay vẫn kéo $W$ về gần 0. Đây là một lý do khiến nhiều bản cài đặt chọn STE thuần thay vì clipped STE.

---

## Phần 6. Lookup Table (LUT) Mixed-Precision GEMM

### 6.1. Nguyên lý cơ bản

Thay vì nhân từng weight với activation, tiền tính tất cả **tổng riêng phần** có thể có của một nhóm activation vào lookup table, rồi dùng weight đã đóng gói làm **chỉ số**.

Cho $g$ weights trong một nhóm, mỗi weight có $C$ giá trị khả dĩ:
- Số tổ hợp: $C^g$.
- Mỗi tổ hợp tương ứng một tổng riêng phần (tính một lần cho mỗi activation vector, dùng lại cho mọi hàng của ma trận weight).

Bản chất của kỹ thuật nằm ở việc chi phí dựng bảng tỷ lệ với $k/g \cdot C^g$ và được chia sẻ cho cả $m$ hàng, trong khi mỗi hàng chỉ còn $k/g$ lần tra bảng và cộng.

### 6.2. TL1: g = 2, C = 3

Ánh xạ trit $w \in \{-1, 0, +1\}$ về chữ số $w + 1 \in \{0, 1, 2\}$ và ghép 2 weights thành một chỉ số 4-bit (Bảng 2 của arXiv:2410.16144):

$$\text{index} = 3\,(w_1 + 1) + (w_2 + 1), \quad \text{index} \in \{0, 1, \ldots, 8\}$$

Ví dụ $(w_1, w_2) = (-1, -1) \mapsto 0$, $(0, 0) \mapsto 4$, $(1, 1) \mapsto 8$. LUT có $3^2 = 9$ entry, mỗi entry chứa:

$$\text{LUT}[\text{index}] = w_1 x_1 + w_2 x_2$$

GEMV cho một hàng:

$$y = \sum_{p=1}^{k/2} \text{LUT}_p[\text{index}_p]$$

Chỉ cần **tra bảng + cộng** (bitnet.cpp dùng LUT int16 và cộng dồn int16).

### 6.3. TL2: g = 3, C = 3

Có $3^3 = 27$ tổ hợp. Vì $\text{LUT}(-w) = -\text{LUT}(w)$, 26 tổ hợp khác 0 chia thành 13 cặp đối nhau; cộng thêm tổ hợp $(0,0,0)$, chỉ cần lưu **14 entry**. Mỗi nhóm 3 weight được mã hóa bằng:

$$\text{code} = (\text{sign}, \text{index}), \quad \text{sign} \in \{0, 1\},\ \text{index} \in \{0, \ldots, 13\}$$

$$w_1 x_1 + w_2 x_2 + w_3 x_3 = (-1)^{\text{sign}} \cdot \text{LUT}[\text{index}]$$

Theo Bảng 3 của paper, $(1,1,1) \mapsto (0, 13)$ còn $(-1,-1,-1) \mapsto (1, 13)$. Tổng cộng 5 bit cho 3 weight:

$$\text{bpw}_{\text{TL2}} = \frac{1 + 4}{3} = \frac{5}{3} \approx 1.67$$

Trong mã sinh bởi `utils/codegen_tl2.py`, bit dấu được lưu trong một mảng riêng và áp dụng bằng chuỗi `_mm256_srai_epi16` / `_mm256_xor_si256` (đảo dấu bù hai không rẽ nhánh) trước khi cộng dồn.

### 6.4. Complexity So sánh

| Phương pháp | Thao tác chính mỗi weight | LUT size mỗi nhóm | Memory |
|:---|:---|:---|:---|
| MAD (I2_S) | unpack 2 bit + int8 multiply-add | - | 2 bpw |
| TL1 (g=2) | 1/2 lần tra bảng + cộng int16 | 9 entries | 2 bpw |
| TL2 (g=3) | 1/3 lần tra bảng + cộng int16 + đảo dấu | 14 entries (nhờ đối xứng) | 1.67 bpw |

TL2 nén tốt nhất và cần ít lần tra bảng nhất trên mỗi weight, nhưng cần xử lý thêm bit dấu và bảng lớn hơn TL1 (14 so với 9 entry). Trade-off cụ thể phụ thuộc vào kích thước cache, số thread và shape ma trận; đó là lý do bitnet.cpp sinh kernel riêng theo shape và có bộ `preset_kernels/` đã tinh chỉnh.

---

## Tóm tắt Công thức Quan trọng

| Công thức | Ý nghĩa |
|:---|:---|
| $\log_2(3) \approx 1.585$ bits | Giới hạn thông tin ternary |
| $54 \cdot 8 / 256 = 1.6875$, $66 \cdot 8 / 256 = 2.0625$ | bpw của TQ1_0 và TQ2_0 |
| $s = \text{mean}(\lvert W \rvert)$ | Absmean scale factor (per-tensor) |
| $\hat{W} = \text{clamp}(\text{round}(W/s), -1, 1)$ | Ternary quantization |
| $P(\hat{W} = 0) \approx 0.31$ với $W$ Gaussian | Tỷ lệ số 0 kỳ vọng |
| $y = \sum_{W=+1} x - \sum_{W=-1} x = \sum t\,x - \sum x$ | Addition-only matmul và mẹo $t = w + 1$ |
| $\text{RMSNorm}(c\,y) = \text{RMSNorm}(y)$ | Scale bị triệt tiêu chỉ khi output đi thẳng vào norm |
| $\partial L/\partial W \approx \partial L/\partial W_q$ | STE thuần |
| $\text{bpw}_{\text{TL2}} = 5/3$ | 1 bit dấu + 4-bit index cho 3 weight |

---

## Tham khảo

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764.
- Wang et al., "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs", arXiv:2410.16144.
- "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs", arXiv:2502.11880.
- Bengio, Léonard, Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation", arXiv:1308.3432, 2013 (nguồn gốc STE).
- Esser et al., "Learned Step Size Quantization", ICLR 2020.
- Mã nguồn: `ggml/src/ggml-common.h` (`block_tq1_0`, `block_tq2_0`), `ggml/src/ggml-quants.c` (`quantize_row_tq1_0_ref`), `ggml/src/ggml-cpu/arch/x86/quants.c` (`ggml_vec_dot_tq2_0_q8_K`), `src/llama-graph.cpp` (`build_lora_mm`).

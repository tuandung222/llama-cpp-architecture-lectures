---
sidebar_position: 8
sidebar_label: "Bài 7: Toán học Ternary Quantization & BitNet b1.58"
---

# Bài 7: Toán học Ternary Quantization & BitNet b1.58

Bài này đi sâu vào toán học đằng sau inference ternary: giới hạn thông tin của weights 3-trạng-thái, phép nhân matrix chỉ dùng addition, hấp thụ scale factors, và kỹ thuật Lookup Table (LUT) cho mixed-precision GEMM.

---

## 1. Giới hạn Thông tin của Ternary Weights

### 1.1. Information-Theoretic Analysis

Mỗi weight ternary nhận một trong 3 giá trị {-1, 0, +1}. Theo Shannon entropy, số bits tối thiểu để mã hóa:

$$H = \log_2(3) \approx 1.585 \text{ bits/weight}$$

Đây là **giới hạn lý thuyết**. Mọi cách đóng gói thực tế đều tốn nhiều hơn:

| Phương pháp | Bits/weight | Overhead |
|:---|:---|:---|
| Giới hạn Shannon | 1.585 | 0% |
| TL2 (3 weights/5 bits) | 1.667 | +5.2% |
| TQ1_0 (llama.cpp) | ~1.69 | +6.6% |
| TQ2_0 (llama.cpp) | ~2.00 | +26.2% |
| I2_S (bitnet.cpp) | 2.000 | +26.2% |
| TL1 (bitnet.cpp) | 2.000 | +26.2% |

TL2 đạt gần giới hạn nhất (chỉ 5.2% overhead) bằng cách pack **3 weights thành 5 bits** ($$3^3 = 27$$ khả dĩ, vừa đủ trong 5 bits = 32 giá trị).

### 1.2. Tại sao 1.58-bit tốt hơn 4-bit?

So sánh memory cho model 7B parameters:

$$M_{\text{FP16}} = 7 \times 10^9 \times 2 = 14 \text{ GB}$$

$$M_{\text{Q4\_K\_M}} = 7 \times 10^9 \times \frac{4.85}{8} \approx 4.2 \text{ GB}$$

$$M_{\text{BitNet}} = 7 \times 10^9 \times \frac{1.67}{8} \approx 1.46 \text{ GB}$$

BitNet tiết kiệm **~10 GB** so với FP16 và **~2.7 GB** so với Q4_K_M.

---

## 2. Absmean Quantization Mathematics

### 2.1. Công thức

Cho weight vector $$W \in \mathbb{R}^n$$, absmean quantization:

$$s = \frac{1}{n} \sum_{i=1}^{n} |W_i| = \frac{\|W\|_1}{n}$$

$$\hat{W}_i = \text{clamp}\!\left(\text{round}\!\left(\frac{W_i}{s}\right), -1, 1\right)$$

### 2.2. So sánh với Symmetric Quantization (Max-Abs)

| Phương pháp | Scale factor | Range |
|:---|:---|:---|
| Symmetric (max-abs) | $$s = \max(\|W\|)$$ | $$W_i/s \in [-1, 1]$$ |
| Absmean | $$s = \text{mean}(\|W\|)$$ | $$W_i/s \in [-n, n]$$ trước clamp |

Absmean cho phép một số outlier weights có magnitude lớn hơn trung bình bị **clamped** về {-1, +1}. Đây là intentional: outlier weights không ảnh hưởng nhiều đến quality khi model được train với quantization constraint.

### 2.3. Threshold Behavior

Với weight distribution chuẩn Gaussian $$W \sim \mathcal{N}(0, \sigma^2)$$:

$$s = \sigma\sqrt{2/\pi}$$

Weight $$|W_i| < s/2$$ sẽ bị round về 0. Khoảng $$\sim 38\%$$ weights sẽ là 0, tạo ra **sparse computation** (bỏ qua 38% operations).

---

## 3. Addition-Only Matrix Multiplication

### 3.1. Proof

Cho $$W \in \{-1, 0, +1\}^{m \times k}$$ và $$x \in \mathbb{R}^k$$:

$$y_i = \sum_{j=1}^{k} W_{ij} \cdot x_j = \sum_{j: W_{ij}=+1} x_j - \sum_{j: W_{ij}=-1} x_j$$

Không có phép nhân nào. Mỗi output element cần tối đa $$k$$ phép cộng/trừ.

### 3.2. Complexity Analysis

| Phương pháp | Multiplications | Additions | Total FLOPs |
|:---|:---|:---|:---|
| FP16 GEMV | $$k$$ | $$k-1$$ | $$2k-1$$ |
| Q4_K (dequant+mul) | $$k$$ (float) | $$k-1$$ | $$2k-1$$ |
| BitNet (addition-only) | **0** | $$\leq k$$ | $$\leq k$$ |

### 3.3. SIMD Implementation

Trên x86 với AVX2/AVX512:

```
// Pseudocode: ternary GEMV
for each 32-element block:
    mask = load_weights()        // {-1, 0, +1} packed
    positive_mask = (mask == +1)
    negative_mask = (mask == -1)
    
    // Add where weight = +1
    acc += masked_add(activations, positive_mask)
    // Subtract where weight = -1  
    acc -= masked_add(activations, negative_mask)
    // Skip where weight = 0 (no operation)
```

Trên ARM NEON, tương tự với `vaddq_s16`/`vsubq_s16` và bitmask.

---

## 4. Scale Absorption vào LayerNorm/RMSNorm

### 4.1. Derivation

BitLinear forward pass:

$$y = \hat{x} \cdot \hat{W} \quad \text{(integer matmul)}$$

$$y_{\text{fp}} = y \times \frac{s_w \cdot s_x}{Q_b} \quad \text{(dequantize)}$$

với $$Q_b = 128$$ (INT8 max).

### 4.2. Absorption

RMSNorm phía sau BitLinear:

$$z = \text{RMSNorm}(y_{\text{fp}}) \cdot \gamma = \frac{y_{\text{fp}}}{\text{RMS}(y_{\text{fp}})} \cdot \gamma$$

Thay $$y_{\text{fp}} = y \cdot c$$ với $$c = s_w \cdot s_x / Q_b$$:

$$z = \frac{y \cdot c}{\text{RMS}(y \cdot c)} \cdot \gamma = \frac{y}{\text{RMS}(y)} \cdot \gamma$$

Hằng số $$c$$ **triệt tiêu** vì RMSNorm là scale-invariant! Do đó:

$$z = \frac{y}{\text{RMS}(y)} \cdot \gamma_{\text{new}}$$

với $$\gamma_{\text{new}}$$ đã absorb mọi scaling. **Zero overhead** tại inference.

### 4.3. So sánh với GGML Dequantization

llama.cpp phải dequantize **mỗi block** trước khi nhân:

```c
// ggml dequantize Q4_0 per block
for each block of 32 values:
    scale = FP16_TO_FP32(block.d)
    for j = 0..31:
        x[j] = (int8_t)(nibble(block.qs[j]) - 8) * scale
```

Chi phí dequantize tích lũy qua mọi layer. BitNet loại bỏ hoàn toàn chi phí này.

---

## 5. Straight-Through Estimator (STE)

### 5.1. Forward Pass

$$W_q = Q(W) = \text{clamp}\!\left(\text{round}\!\left(\frac{W}{s}\right), -1, 1\right)$$

Hàm $$Q$$ không khả vi (step function), gradient không tồn tại.

### 5.2. Backward Pass: STE Approximation

$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q} \cdot \mathbf{1}_{|W/s| < 1}$$

Gradient đi qua như identity khi weight nằm trong clipping range, và bằng 0 khi bị clip. Đây là **ước lượng đơn giản nhất**, không cố gắng tính đúng gradient của hàm step.

### 5.3. Gradient Clipping Effect

| Region | Gradient | Effect |
|:---|:---|:---|
| $$\|W/s\| < 1$$ | Normal | Weight được optimize bình thường |
| $$\|W/s\| \geq 1$$ | Zero | Weight bị "đóng băng", không update |

Weights bị đóng băng là **acceptable** vì chúng đã ở giá trị tối ưu (+1 hoặc -1).

---

## 6. Lookup Table (LUT) Mixed-Precision GEMM

### 6.1. Nguyên lý cơ bản

Thay vì dequantize weights rồi nhân, pre-compute tất cả **partial sums** vào lookup table.

Cho $$g$$ weights trong một group, mỗi weight có $$C$$ giá trị khả dĩ:
- Số combinations: $$C^g$$
- Mỗi combination -> một partial sum (pre-computed)

### 6.2. TL1: Element-wise LUT (g=2, C=3)

Pack 2 weights ternary ($$C=3$$) thành một 4-bit index:

$$\text{index} = w_1 \times 3 + w_2, \quad \text{index} \in \{0, 1, ..., 8\}$$

LUT có $$3^2 = 9$$ entries. Mỗi entry chứa:

$$\text{LUT}[\text{index}] = w_1 \cdot x_1 + w_2 \cdot x_2$$

GEMV cho toàn bộ vector:

$$y = \sum_{k=1}^{K/g} \text{LUT}_k[\text{index}_k]$$

Chỉ cần **lookup + addition**, không nhân.

### 6.3. TL2: 3-weight Packing (g=3, C=3)

Pack 3 weights thành 5-bit index ($$3^3 = 27 \leq 32$$):

$$\text{index} = w_1 \times 9 + w_2 \times 3 + w_3, \quad \text{index} \in \{0, ..., 26\}$$

LUT có 27 entries. Compression: $$\frac{3 \times 2}{5} = 1.67 \text{ bpw}$$.

### 6.4. Complexity So sánh

| Phương pháp | Ops per element | LUT size | Memory |
|:---|:---|:---|:---|
| MAD (I2_S) | 1 unpack + 1 add | - | 2 bpw |
| TL1 (g=2) | 1 lookup + 1 add | 9 entries | 2 bpw |
| TL2 (g=3) | 1 lookup + 1 add | 27 entries | 1.67 bpw |

TL2 đạt compression tốt nhất nhưng LUT lớn hơn (27 vs 9 entries). Trade-off phụ thuộc vào CPU cache size.

---

## Tóm tắt Công thức Quan trọng

| Công thức | Ý nghĩa |
|:---|:---|
| $$\log_2(3) = 1.585$$ bits | Giới hạn thông tin ternary |
| $$s = \text{mean}(\|W\|)$$ | Absmean scale factor |
| $$\hat{W} = \text{clamp}(\text{round}(W/s), -1, 1)$$ | Ternary quantization |
| $$y = \sum_{W=+1} x - \sum_{W=-1} x$$ | Addition-only matmul |
| $$\gamma_{\text{new}} = \gamma_{\text{RMSNorm}} \cdot s_w$$ | Scale absorption |
| $$\partial L/\partial W \approx \partial L/\partial W_q \cdot \mathbf{1}_{\|W/s\|<1}$$ | STE gradient |

---

## Tham khảo

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764.
- Wang et al., "1-bit AI Infra: Fast and Lossless BitNet b1.58 Inference on CPUs", arXiv:2410.16144.
- Esser et al., "Learned Step Size Quantization", ICLR 2020 (STE foundation).

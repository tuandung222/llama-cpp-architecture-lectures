---
sidebar_position: 2
sidebar_label: "Bài 1: Toán học Quantization Error & Information Loss"
---

# Bài 1: Toán học Quantization Error & Information Loss

Bài này phân tích cơ sở toán học của quantization error: tại sao nén trọng số gây mất mát chất lượng, và làm thế nào để đo lường, dự đoán, và tối thiểu hóa mất mát đó.

---

## 1. Uniform Quantization Error

Với uniform quantization step $\Delta$, quantization error $\epsilon = x - \hat{x}$ tuân theo phân phối đều:

$$\epsilon \sim \text{Uniform}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$$

Mean Squared Error (MSE):

$$MSE = \mathbb{E}[\epsilon^2] = \frac{\Delta^2}{12}$$

Đọc công thức này: MSE tỷ lệ với bình phương của step size. Giảm step size (tăng số bit) sẽ giảm MSE theo hàm bậc hai.

---

## 2. Signal-to-Quantization-Noise Ratio (SQNR)

$$SQNR = 10 \log_{10}\left(\frac{\sigma_x^2}{\sigma_\epsilon^2}\right) = 6.02b + 1.76 \text{ dB}$$

Trong đó $b$ là số bit quantization và $\sigma_x^2$ là variance của tín hiệu gốc.

**Hệ quả**: Mỗi bit thêm vào tăng SQNR ~6 dB (tương đương giảm MSE ~4x).

| Quant | Bits | SQNR | MSE ratio vs FP16 |
|:---|:---|:---|:---|
| Q8_0 | 8 | 49.9 dB | ~1x (baseline) |
| Q5_K | 5.5 | 34.9 dB | ~32x |
| Q4_K | 4.5 | 28.9 dB | ~128x |
| Q3_K | 3.4 | 22.2 dB | ~512x |

---

## 3. Lloyd-Max Algorithm

Lloyd-Max tìm optimal quantization levels cho một phân phối xác suất đã biết:

**Input**: PDF $f(x)$ của trọng số, số mức $L = 2^b$.

**Iterate**:
1. Cho trước các mức $y_i$, tìm boundaries: $x_i = \frac{y_i + y_{i+1}}{2}$
2. Cho trước boundaries $x_i$, tìm optimal levels: $y_i = \frac{\int_{x_{i-1}}^{x_i} x f(x) dx}{\int_{x_{i-1}}^{x_i} f(x) dx}$

Lặp đến khi hội tụ. Lloyd-Max cho biết optimal quantization levels KHÔNG đều khi phân phối trọng số không đều (thường là Gaussian).

---

## 4. Block-wise Quantization Error Propagation

Trong transformer, quantization error propagate qua các layers:

$$\text{Output Error} \approx \sum_{l=1}^{L} \prod_{j=l+1}^{L} \|W_j\|_2 \cdot \|\Delta W_l\|_2 \cdot \|h_{l-1}\|_2$$

Trong đó $\Delta W_l$ là quantization error của weight matrix layer $l$.

**Hệ quả thực tế**: Layers gần output (cuối model) ít ảnh hưởng bởi quantization hơn layers gần input. Đây là lý do Q4_K_M giữ attention layers ở FP16.

---

## 5. Importance Matrix và Optimal Bit Allocation

Given a budget of $B$ total bits for $N$ weight groups, optimal allocation:

$$b_i = \bar{b} + \frac{1}{2} \log_2\left(\frac{\sigma_i^2}{\prod_j \sigma_j^{2/N}}\right)$$

Trong đó $\sigma_i^2$ là variance của group $i$ và $\bar{b} = B/N$. Đây chính là cơ sở toán học của **importance matrix quantization** (I-quants) trong llama.cpp.

---

## 💡 Đúc kết

Toán học quantization cho thấy:
- SQNR giảm ~6 dB cho mỗi bit mất đi.
- Lloyd-Max tối ưu cho phân phối không đều.
- Error propagate tích lũy qua layers, nhưng layers cuối ít bị ảnh hưởng hơn.
- Importance-based allocation (I-quants) đạt tối ưu theoretical bound.

---
sidebar_position: 2
sidebar_label: "Case 1: Q4_K_M vs Q5_K_M vs Q8_0"
---

# Case 1: So sánh Q4_K_M vs Q5_K_M vs Q8_0

## 1. Bài toán thực tế

Khi deploy một mô hình 8B tham số, bạn cần chọn quant type phù hợp. Ba lựa chọn phổ biến nhất: Q4_K_M, Q5_K_M, và Q8_0. Case study này so sánh toàn diện ba quant trên các tiêu chí: chất lượng, tốc độ, và bộ nhớ.

## 2. Test Methodology

- **Model**: Llama-3-8B-Instruct
- **Hardware**: Ryzen 7 5800X (AVX2), RTX 3090 (CUDA)
- **Benchmarks**: WikiText-2 PPL, llama-bench throughput

## 3. Kết quả

| Quant | Size | PPL (WikiText-2) | Δ PPL | CPU tok/s | GPU tok/s |
|:---|:---|:---|:---|:---|:---|
| FP16 | 16 GB | 6.14 | 0.00 | 3.1 | 120 |
| Q8_0 | 8.5 GB | 6.15 | +0.01 | 6.0 | 85 |
| Q5_K_M | 5.7 GB | 6.20 | +0.06 | 8.5 | 78 |
| **Q4_K_M** | **4.9 GB** | **6.35** | **+0.21** | **10.2** | **80** |
| Q3_K_M | 4.0 GB | 6.72 | +0.58 | 12.5 | 82 |

## 4. Phân tích

- **Q8_0**: Gần như lossless, nhưng chỉ giảm 50% kích thước. Phù hợp khi RAM/VRAM dư dả.
- **Q5_K_M**: Sweet spot cho quality-first. PPL tăng chỉ 1% so với FP16.
- **Q4_K_M**: Sweet spot cho balanced. PPL tăng 3.4%, giảm 70% kích thước, tăng 3.3x CPU throughput.
- **Q3_K_M**: Chấp nhận được cho experimentation, PPL tăng 9.4%.

## 5. Khuyến nghị

| Use case | Recommended Quant |
|:---|:---|
| Production (quality-critical) | Q8_0 hoặc Q6_K |
| Production (balanced) | **Q4_K_M** hoặc Q5_K_M |
| Edge/IoT | Q3_K_M hoặc IQ4_XS |
| Research/prototyping | Q2_K |

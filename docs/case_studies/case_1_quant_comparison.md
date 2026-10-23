---
sidebar_position: 2
sidebar_label: "Case 1: Q4_K_M vs Q5_K_M vs Q8_0"
---

# Case 1: So sánh Q4_K_M vs Q5_K_M vs Q8_0

## Phần 1. Bài toán thực tế

Khi deploy một mô hình 8B tham số, bạn cần chọn quant type phù hợp. Ba lựa chọn phổ biến nhất: Q4_K_M, Q5_K_M, và Q8_0. Case study này so sánh ba quant trên các tiêu chí: chất lượng, tốc độ, và bộ nhớ, đồng thời chỉ ra cách tự đo lại trên phần cứng của bạn.

## Phần 2. Test Methodology

- **Model**: Meta-Llama-3-8B (bản base; bản Instruct có PPL trên WikiText-2 cao hơn đáng kể nên không so sánh trực tiếp được với các con số dưới đây).
- **Hardware**: Ryzen 7 5800X (AVX2, DDR4-3200 hai kênh, khoảng 51.2 GB/s), RTX 3090 (khoảng 936 GB/s).
- **Benchmarks**: WikiText-2 PPL bằng `llama-perplexity`, throughput bằng `llama-bench`.

```bash
# Chuẩn bị dữ liệu (script có sẵn trong repo llama.cpp)
./scripts/get-wikitext-2.sh            # tạo wikitext-2-raw/wiki.test.raw

# Chất lượng
./build/bin/llama-perplexity -m llama-3-8b-q4_k_m.gguf -f wikitext-2-raw/wiki.test.raw -c 512

# Tốc độ: pp512 (prefill) và tg128 (decode), 5 lần lặp mặc định
./build/bin/llama-bench -m llama-3-8b-f16.gguf,llama-3-8b-q8_0.gguf,llama-3-8b-q4_k_m.gguf -p 512 -n 128
```

## Phần 3. Kết quả

Vì pha decode bị giới hạn bởi băng thông bộ nhớ, trần lý thuyết của tốc độ sinh token là:

$$\text{tok/s}_{\max} \approx \frac{B}{S}$$

với $B$ là memory bandwidth (GB/s) và $S$ là số byte weight phải đọc cho mỗi token (xấp xỉ kích thước file). Thực đo thường đạt khoảng 60% đến 85% trần này.

| Quant | Size | PPL (WikiText-2, c=512) | Δ PPL | Trần CPU (51.2 GB/s) | Trần GPU (936 GB/s) |
|:---|:---|:---|:---|:---|:---|
| FP16 | 16.1 GB | 6.14 | 0.00 | ~3.2 tok/s | ~58 tok/s |
| Q8_0 | 8.5 GB | 6.15 | +0.01 | ~6.0 tok/s | ~110 tok/s |
| Q5_K_M | 5.7 GB | 6.20 | +0.06 | ~9.0 tok/s | ~164 tok/s |
| **Q4_K_M** | **4.9 GB** | **6.35** | **+0.21** | **~10.4 tok/s** | **~191 tok/s** |
| Q3_K_M | 4.0 GB | 6.72 | +0.58 | ~12.8 tok/s | ~234 tok/s |

Các giá trị PPL là số liệu minh họa điển hình: PPL tuyệt đối phụ thuộc vào `-c`, phiên bản tokenizer và dữ liệu, nên chỉ so sánh các dòng đo **cùng một cấu hình**. Trên GPU, các kiểu K-quant bit thấp còn chịu chi phí giải mã trong kernel, nên khoảng cách thực đo giữa Q4_K_M và Q3_K_M thường nhỏ hơn tỷ lệ kích thước gợi ý.

## Phần 4. Phân tích

- **Q8_0**: gần như lossless, kích thước khoảng 53% so với FP16 (giảm khoảng 47%). Phù hợp khi RAM/VRAM dư dả.
- **Q5_K_M**: sweet spot cho quality-first. PPL tăng khoảng 1% so với FP16.
- **Q4_K_M**: sweet spot cho balanced. PPL tăng khoảng 3.4%, giảm khoảng 69% kích thước, trần throughput decode tăng $16.1 / 4.9 \approx 3.3$ lần.
- **Q3_K_M**: chấp nhận được cho thử nghiệm, PPL tăng khoảng 9.4%.

PPL là thước đo trung bình; để thấy sai lệch trên từng token, nên dùng thêm KL divergence so với model FP16:

```bash
# Lưu logits của model gốc một lần
./build/bin/llama-perplexity -m llama-3-8b-f16.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base logits-f16.bin
# So sánh từng bản quant với logits gốc
./build/bin/llama-perplexity -m llama-3-8b-q4_k_m.gguf --kl-divergence-base logits-f16.bin --kl-divergence
```

## Phần 5. Khuyến nghị

| Use case | Recommended Quant |
|:---|:---|
| Production (quality-critical) | Q8_0 hoặc Q6_K |
| Production (balanced) | **Q4_K_M** hoặc Q5_K_M |
| Edge/IoT | IQ4_XS (với imatrix) hoặc Q3_K_M |
| Research/prototyping | Q2_K hoặc IQ2/IQ3 (bắt buộc hoặc rất nên có imatrix) |

Lưu ý: các kiểu bit cực thấp mới trong upstream như `TQ1_0`/`TQ2_0` (ternary) và `Q1_0`/`Q2_0` được thiết kế cho model **vốn đã** được huấn luyện hoặc chuyển đổi về miền 1-bit/ternary (BitNet, Bonsai), không phải để PTQ trực tiếp một model như Llama 3; xem [Bài 10](../lesson_10_bitnet_cpp) và [Bài 11](../lesson_11_bonsai_model_family).

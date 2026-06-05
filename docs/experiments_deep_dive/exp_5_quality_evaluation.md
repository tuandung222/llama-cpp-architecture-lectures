# Bài 5: Perplexity Evaluation - Đánh giá chất lượng Quant

## 1. Mục tiêu

Sử dụng `llama-perplexity` để đo perplexity (PPL) của nhiều quantization variant trên cùng một benchmark dataset. Kết quả dùng để chọn quant tối ưu: cân bằng giữa kích thước file và chất lượng output.

---

## 2. Lý thuyết nhanh

Perplexity đo "độ ngạc nhiên" của model với text chưa thấy:

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(x_i | x_{\lt i})\right)$$

- PPL thấp = model dự đoán tốt = chất lượng cao.
- FP16 baseline thường có PPL thấp nhất.
- Quantization làm tăng PPL (giảm chất lượng).

---

## 3. Chạy Perplexity Test

### Cơ bản

```bash
./llama-perplexity \
    -m model-q4_k_m.gguf \
    -f wikitext-2-raw.txt \
    -c 512 \
    -t 8 \
    --chunks 128
```

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model file cần test |
| `-f` | Evaluation dataset (raw text) |
| `-c` | Context length per chunk |
| `--chunks` | Số chunk để đánh giá |

### Chuẩn bị dataset

```bash
# Tải WikiText-2 (standard benchmark)
wget https://huggingface.co/datasets/ggml-org/wikitext-2-raw-v1/resolve/main/wikitext-2-raw-v1.txt

# Hoặc chuẩn bị custom dataset
# File text thuần, mỗi dòng một câu, tối thiểu 100K tokens
```

---

## 4. So sánh toàn diện nhiều Quant

```bash
#!/bin/bash
# benchmark_ppl.sh - So sánh PPL giữa các quant

MODELS=(
    "model-f16.gguf"
    "model-q8_0.gguf"
    "model-q6_k.gguf"
    "model-q5_k_m.gguf"
    "model-q5_k_s.gguf"
    "model-q4_k_m.gguf"
    "model-q4_k_s.gguf"
    "model-q3_k_m.gguf"
    "model-q2_k.gguf"
)

DATASET="wikitext-2-raw-v1.txt"
CTX=512
CHUNKS=128
THREADS=8

echo "| Quant | Size (GB) | PPL | Delta PPL |"
echo "|-------|-----------|-----|-----------|"

for model in "${MODELS[@]}"; do
    size=$(du -h "$model" | cut -f1)
    ppl=$(./llama-perplexity \
        -m "$model" \
        -f "$DATASET" \
        -c $CTX \
        -t $THREADS \
        --chunks $CHUNKS 2>&1 | \
        grep "Final estimate" | \
        grep -oP 'PPL = \K[0-9.]+')
    echo "| $(basename $model .gguf) | $size | $ppl | TBD |"
done
```

### Kết quả mẫu (Llama 3 8B trên WikiText-2)

| Quant | Size (GB) | PPL | Delta vs FP16 |
|:---|:---|:---|:---|
| F16 | 15.2 | 6.14 | 0.00 |
| Q8_0 | 8.1 | 6.18 | +0.04 |
| Q6_K | 6.2 | 6.21 | +0.07 |
| Q5_K_M | 5.4 | 6.25 | +0.11 |
| Q5_K_S | 5.2 | 6.29 | +0.15 |
| Q4_K_M | 4.6 | 6.38 | +0.24 |
| Q4_K_S | 4.3 | 6.47 | +0.33 |
| Q3_K_M | 3.6 | 6.72 | +0.58 |
| Q2_K | 2.8 | 7.35 | +1.21 |

---

## 5. Phân tích kết quả

### Vùng "sweet spot"

Dựa trên kết quả PPL, có thể chia làm 3 vùng:

1. **Near-lossless** (Delta PPL < 0.1): Q8_0, Q6_K. Phù hợp cho production cần chất lượng cao nhất.
2. **Balanced** (Delta PPL 0.1-0.3): Q5_K_M, Q5_K_S, Q4_K_M. Best trade-off size vs quality.
3. **Aggressive** (Delta PPL > 0.5): Q3_K_M, Q2_K. Chỉ dùng khi bộ nhớ rất hạn chế.

### Khuyến nghị theo use case

| Use case | Quant khuyến nghị | Lý do |
|:---|:---|:---|
| Production server (RAM đủ) | Q6_K hoặc Q8_0 | Chất lượng gần FP16 |
| Production server (RAM hạn chế) | Q4_K_M hoặc Q5_K_M | Best trade-off |
| Local chatbot | Q4_K_M | Vừa RAM laptop, chất lượng OK |
| Edge/mobile | Q4_0 hoặc Q3_K_M | Tối thiểu bộ nhớ |
| Research/evaluation | Q8_0 | Chuẩn so sánh |

---

## 6. Context Length Impact

Context length ảnh hưởng đến PPL measurement:

```bash
# Test PPL ở nhiều context length
for ctx in 128 256 512 1024 2048; do
    ppl=$(./llama-perplexity \
        -m model-q4_k_m.gguf \
        -f wikitext.txt \
        -c $ctx -t 8 --chunks 64 2>&1 | \
        grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')
    echo "ctx=$ctx PPL=$ppl"
done
```

PPL thường giảm nhẹ khi tăng context (model có nhiều context hơn để dự đoán), nhưng thời gian benchmark tăng tuyến tính.

---

## 7. Chunk Count Impact

Số chunk càng nhiều, PPL estimate càng chính xác (variance giảm):

| Chunks | Thời gian (phút) | PPL Variance |
|:---|:---|:---|
| 32 | ~5 | +/- 0.08 |
| 64 | ~10 | +/- 0.04 |
| 128 | ~20 | +/- 0.02 |
| 256 | ~40 | +/- 0.01 |

Khuyến nghị: tối thiểu 128 chunks cho báo cáo nghiêm túc, 64 chunks cho quick check.

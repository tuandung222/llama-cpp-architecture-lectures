# Bài 5: Perplexity Evaluation - Đánh giá chất lượng Quant

## Phần 1. Mục tiêu

Sử dụng `llama-perplexity` để đo perplexity (PPL) của nhiều quantization variant trên cùng một benchmark dataset. Kết quả dùng để chọn quant tối ưu: cân bằng giữa kích thước file và chất lượng output.

---

## Phần 2. Lý thuyết nhanh

Perplexity đo "độ ngạc nhiên" của model với text chưa thấy:

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(x_i | x_{\lt i})\right)$$

- PPL thấp = model dự đoán tốt = chất lượng cao.
- FP16/BF16 baseline thường có PPL thấp nhất.
- Quantization làm tăng PPL (giảm chất lượng).
- PPL chỉ so sánh được giữa các model **cùng tokenizer**, cùng dữ liệu và cùng `-c`.

`llama-perplexity` chia văn bản thành các chunk dài `-c` token và, với mỗi chunk, chỉ tính log-likelihood trên **nửa sau** của chunk (nửa đầu đóng vai trò context), nên mỗi token được chấm điểm khi đã có ít nhất $c/2$ token ngữ cảnh.

---

## Phần 3. Chạy Perplexity Test

### Cơ bản

```bash
./build/bin/llama-perplexity \
    -m model-q4_k_m.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 512 \
    -t 8 \
    --chunks 128
```

Dòng cuối của output có dạng `Final estimate: PPL = 6.3812 +/- 0.03921`; phần `+/-` là sai số chuẩn ước lượng từ dữ liệu.

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model file cần test |
| `-f` | Evaluation dataset (raw text) |
| `-c` | Context length per chunk |
| `--chunks` | Số chunk để đánh giá |

### Chuẩn bị dataset

```bash
# Tải WikiText-2 bằng script của repo llama.cpp (cần wget/curl và unzip)
./scripts/get-wikitext-2.sh
# -> wikitext-2-raw/wiki.test.raw (dùng để đánh giá), wiki.train.raw (dùng để calibrate imatrix)

# Hoặc chuẩn bị custom dataset: file text thuần, tối thiểu khoảng 100K tokens.
# llama-perplexity yêu cầu file có ít nhất 2 x c token.
```

---

## Phần 4. So sánh toàn diện nhiều Quant

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

DATASET="wikitext-2-raw/wiki.test.raw"
CTX=512
CHUNKS=128
THREADS=8

echo "| Quant | Size (GB) | PPL | Delta PPL |"
echo "|-------|-----------|-----|-----------|"

for model in "${MODELS[@]}"; do
    size=$(du -h "$model" | cut -f1)
    ppl=$(./build/bin/llama-perplexity \
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
| F16 | 14.96 | 6.14 | 0.00 |
| Q8_0 | 7.95 | 6.18 | +0.04 |
| Q6_K | 6.14 | 6.21 | +0.07 |
| Q5_K_M | 5.33 | 6.25 | +0.11 |
| Q5_K_S | 5.21 | 6.29 | +0.15 |
| Q4_K_M | 4.58 | 6.38 | +0.24 |
| Q4_K_S | 4.37 | 6.47 | +0.33 |
| Q3_K_M | 3.74 | 6.72 | +0.58 |
| Q2_K | 2.96 | 7.35 | +1.21 |

Kích thước tính bằng GiB (như `llama-bench` báo); các giá trị PPL là số liệu minh họa xu hướng. Lưu ý: `du -h` trong script dùng đơn vị nhị phân (G = GiB).

---

## Phần 5. Phân tích kết quả

### Vùng "sweet spot"

Dựa trên kết quả PPL, có thể chia làm 3 vùng:

1. **Near-lossless** (Delta PPL dưới 0.1): Q8_0, Q6_K. Phù hợp cho production cần chất lượng cao nhất.
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

## Phần 6. Context Length Impact

Context length ảnh hưởng đến PPL measurement:

```bash
# Test PPL ở nhiều context length
for ctx in 128 256 512 1024 2048; do
    ppl=$(./build/bin/llama-perplexity \
        -m model-q4_k_m.gguf \
        -f wikitext-2-raw/wiki.test.raw \
        -c $ctx -t 8 --chunks 64 2>&1 | \
        grep "Final estimate" | grep -oP 'PPL = \K[0-9.]+')
    echo "ctx=$ctx PPL=$ppl"
done
```

PPL thường giảm khi tăng context (mỗi token được chấm điểm với nhiều ngữ cảnh hơn). Với số chunk cố định, số token được đánh giá tăng tỷ lệ với `-c` và chi phí attention mỗi token cũng tăng, nên thời gian chạy tăng **nhanh hơn tuyến tính** theo `-c`. Vì PPL phụ thuộc `-c`, chỉ so sánh các quant ở cùng một giá trị context.

---

## Phần 7. Chunk Count Impact

Số chunk càng nhiều, PPL estimate càng ổn định. Gọi $\ell_t = -\log p(x_t \mid x_{\lt t})$; PPL là $\exp(\bar{\ell})$ với $\bar{\ell}$ là trung bình trên $N$ token được chấm điểm. Nếu coi các $\ell_t$ gần độc lập với phương sai $\sigma_\ell^2$, theo khai triển delta:

$$\text{SE}(\text{PPL}) \approx \text{PPL} \cdot \frac{\sigma_\ell}{\sqrt{N}}$$

Sai số chuẩn giảm theo $1/\sqrt{N}$, **không** theo $1/N$: gấp đôi số chunk chỉ giảm sai số khoảng $\sqrt{2} \approx 1.41$ lần.

| Chunks | Thời gian (tương đối) | Sai số chuẩn PPL (minh họa) |
|:---|:---|:---|
| 32 | 1x | +/- 0.080 |
| 64 | 2x | +/- 0.057 |
| 128 | 4x | +/- 0.040 |
| 256 | 8x | +/- 0.028 |

Khi so sánh hai quant **trên cùng dữ liệu**, sai số của hiệu số nhỏ hơn nhiều so với sai số của từng PPL (vì các token khó/dễ là chung), nên KL divergence hoặc hiệu PPL theo cặp nhạy hơn so sánh hai khoảng tin cậy độc lập:

```bash
./build/bin/llama-perplexity -m model-f16.gguf -f wikitext-2-raw/wiki.test.raw --kl-divergence-base logits-f16.bin
./build/bin/llama-perplexity -m model-q4_k_m.gguf --kl-divergence-base logits-f16.bin --kl-divergence
```

Khuyến nghị: chạy toàn bộ `wiki.test.raw` cho báo cáo nghiêm túc (bỏ `--chunks`), 64 đến 128 chunk cho quick check. Ngoài PPL, `llama-perplexity` còn chấm các benchmark trắc nghiệm qua `--hellaswag`, `--winogrande`, `--multiple-choice` (dữ liệu tải bằng `scripts/get-hellaswag.sh`, `scripts/get-winogrande.sh`).

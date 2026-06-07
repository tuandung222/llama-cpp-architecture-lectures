# Bài 2: Custom Quantization với Importance Matrix

## 1. Mục tiêu

Tạo importance matrix (imatrix) từ calibration data, sau đó dùng nó để quantize model đạt chất lượng cao hơn default quantization. Imatrix quantization (I-quants) giữ lại nhiều thông tin hơn ở các weight quan trọng.

---

## 2. Lý thuyết nền

### Tại sao default quantization không tối ưu?

Default quantization (Q4_K_M, Q5_K_M...) áp dụng cùng một strategy cho tất cả weight. Tuy nhiên, không phải weight nào cũng quan trọng như nhau. Một số output channel của attention layer có ảnh hưởng lớn đến chất lượng model, trong khi một số khác có thể bị quantize mạnh hơn mà không ảnh hưởng đáng kể.

### Importance Matrix là gì?

Importance matrix $H$ đo "độ quan trọng" của mỗi weight element dựa trên activation statistics:

$$H_{ij} = \mathbb{E}_{x \sim \mathcal{D}} \left[ x_i^2 \right] \cdot |W_{ij}|$$

Trong đó $x_i$ là input activation tại vị trí $i$, lấy trung bình trên calibration dataset $\mathcal{D}$. Weight có $H_{ij}$ cao được quantize chính xác hơn (nhiều bit hơn), weight có $H_{ij}$ thấp được quantize mạnh hơn.

---

## 3. Step-by-step: Tạo Imatrix

### Bước 1: Chuẩn bị calibration data

```bash
# Tạo file text calibration (tối thiểu 1-2 MB text)
# Dùng dữ liệu cùng domain với use case thực tế
cat > calibration_data.txt << 'EOF'
The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Quantization reduces model size while maintaining quality...
EOF

# Hoặc dùng dataset có sẵn
wget https://huggingface.co/datasets/ggml-org/imatrix-data/resolve/main/wikitext-2-raw-v1.txt
```

### Bước 2: Chạy imatrix collection

```bash
./llama-imatrix \
    -m model-f16.gguf \
    -f calibration_data.txt \
    -of imatrix.dat \
    -c 512 \
    -t 4 \
    --chunks 256
```

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model FP16 hoặc Q8_0 (cần đủ chính xác để collect stats) |
| `-f` | File calibration data |
| `-of` | Output imatrix file |
| `-c` | Context length cho mỗi chunk |
| `--chunks` | Số chunk để lấy trung bình (càng nhiều càng chính xác) |

Output: file `imatrix.dat` chứa importance scores cho mỗi weight tensor.

---

## 4. Quantize với Imatrix

```bash
# Quantize Q4_K_M với imatrix (I-quant)
./llama-quantize \
    --imatrix imatrix.dat \
    model-f16.gguf \
    model-iq4_xs.gguf \
    IQ4_XS
```

### Các I-quant types phổ biến

| Quant | Bits/param | Kích thước tương đối | Chất lượng |
|:---|:---|:---|:---|
| IQ2_XXS | 2.06 | Rất nhỏ | Thấp |
| IQ2_XS | 2.31 | Rất nhỏ | Thấp-trung bình |
| IQ3_XXS | 3.06 | Nhỏ | Trung bình |
| IQ3_S | 3.44 | Nhỏ | Khá |
| IQ4_XS | 4.25 | Nhỏ-vừa | Tốt |
| IQ4_NL | 4.50 | Vừa | Rất tốt |

---

## 5. So sánh: Default vs Imatrix Quant

```bash
# Default Q4_K_M
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Imatrix IQ4_XS
./llama-quantize --imatrix imatrix.dat \
    model-f16.gguf model-iq4_xs.gguf IQ4_XS

# So sánh perplexity
./llama-perplexity -m model-q4_k_m.gguf -f wikitext.txt -c 512
./llama-perplexity -m model-iq4_xs.gguf -f wikitext.txt -c 512
```

### Kết quả kỳ vọng

| Model | PPL (thấp hơn = tốt hơn) | Size |
|:---|:---|:---|
| FP16 (baseline) | 5.82 | 100% |
| Q4_K_M (default) | 6.25 | ~30% |
| IQ4_XS (imatrix) | 6.10 | ~30% |
| Q8_0 (default) | 5.87 | ~50% |

Imatrix quant cải thiện PPL khoảng 0.1-0.3 điểm so với default quant cùng kích thước, đặc biệt rõ rệt ở các quant thấp (2-4 bit).

---

## 6. Best Practices

1. **Calibration data chất lượng**: dùng text cùng domain với use case (code data cho code model, Vietnamese data cho Vietnamese model).
2. **Đủ chunks**: tối thiểu 256 chunks, lý tưởng 512-1024.
3. **Model đủ chính xác**: dùng FP16 hoặc Q8_0 để collect imatrix, không dùng Q4.
4. **Context length phù hợp**: 512-2048 tokens, tương ứng context length thực tế.

---

## 7. Troubleshooting

| Vấn đề | Nguyên nhân | Fix |
|:---|:---|:---|
| Imatrix file quá nhỏ | Chunks quá ít | Tăng `--chunks` lên 512+ |
| PPL không cải thiện | Calibration data khác domain | Dùng data phù hợp hơn |
| OOM khi collect imatrix | Model quá lớn cho RAM | Dùng Q8_0 thay vì FP16 |
| Imatrix không tương thích | Model khác version | Collect lại imatrix cho đúng model |

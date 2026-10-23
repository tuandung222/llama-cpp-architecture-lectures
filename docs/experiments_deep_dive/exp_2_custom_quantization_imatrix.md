# Bài 2: Custom Quantization với Importance Matrix

## Phần 1. Mục tiêu

Tạo importance matrix (imatrix) từ calibration data, sau đó dùng nó để quantize model đạt chất lượng cao hơn quantization không có imatrix. Imatrix dùng được cho cả K-quants (Q4_K_M, Q3_K_S...) lẫn I-quants (IQ4_XS, IQ2_XXS...), và là **bắt buộc** với các kiểu bit rất thấp: `tensor_requires_imatrix()` trong `src/llama-quant.cpp` yêu cầu imatrix cho IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS (và Q2_K khi tạo file Q2_K_S), nếu thiếu `llama-quantize` dừng với lỗi "this quantization requires an importance matrix".

---

## Phần 2. Lý thuyết nền

### Tại sao default quantization không tối ưu?

Khi không có imatrix, các hàm quantize của GGML chọn scale (và min) của mỗi block để cực tiểu sai số bình phương trên **weight**, coi mọi phần tử như nhau. Nhưng thứ ta thực sự quan tâm là sai số trên **output** $y = W x$. Các cột của $W$ ứng với kênh input có activation lớn đóng góp nhiều hơn vào sai số output, nên cần được xấp xỉ chính xác hơn.

### Importance Matrix là gì?

Xét một hàng $w \in \mathbb{R}^k$ của $W$ và bản lượng tử $\hat{w}$. Sai số output kỳ vọng trên dữ liệu calibration $\mathcal{D}$:

$$\mathbb{E}_{x \sim \mathcal{D}}\left[\big((w - \hat{w})^\top x\big)^2\right] = (w - \hat{w})^\top \, \mathbb{E}[x x^\top] \, (w - \hat{w}) \approx \sum_{j=1}^{k} I_j \, (w_j - \hat{w}_j)^2$$

với xấp xỉ đường chéo (bỏ qua tương quan giữa các kênh) và:

$$I_j = \mathbb{E}_{x \sim \mathcal{D}}\left[x_j^2\right] \approx \frac{1}{N} \sum_{t=1}^{N} x_{t,j}^2$$

$I_j$ là "độ quan trọng" của **kênh input** $j$, dùng chung cho mọi hàng của ma trận; `llama-imatrix` thu thập đúng đại lượng này (một vector cho mỗi tensor weight, tách riêng theo expert với MoE) bằng cách quan sát activation đi vào mỗi phép `mul_mat`. Khi quantize, các hàm `quantize_*` dùng $I_j$ (thường kết hợp với độ lớn của chính weight) làm **trọng số trong weighted MSE** để chọn scale và giá trị lượng tử của block. Imatrix **không** cấp thêm bit cho weight quan trọng: số bit mỗi weight vẫn do quant type quyết định, chỉ là sai số được phân bổ lại sao cho kênh quan trọng chịu sai số nhỏ hơn.

---

## Phần 3. Step-by-step: Tạo Imatrix

### Bước 1: Chuẩn bị calibration data

```bash
# Tạo file text calibration (tối thiểu 1-2 MB text)
# Dùng dữ liệu cùng domain với use case thực tế
cat > calibration_data.txt << 'EOF'
The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Quantization reduces model size while maintaining quality...
EOF

# Hoặc dùng tập train của WikiText-2 (script có sẵn trong repo llama.cpp)
./scripts/get-wikitext-2.sh    # tạo wikitext-2-raw/wiki.train.raw, wiki.valid.raw, wiki.test.raw
# Dùng wiki.train.raw để calibrate; giữ wiki.test.raw cho đánh giá PPL ở Bài 5,
# tránh calibrate và đánh giá trên cùng dữ liệu.
```

### Bước 2: Chạy imatrix collection

```bash
./build/bin/llama-imatrix \
    -m model-f16.gguf \
    -f wikitext-2-raw/wiki.train.raw \
    -o imatrix.gguf \
    -c 512 \
    -t 4 \
    --chunks 256 \
    -ngl 99
```

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model F16/BF16 hoặc Q8_0 (cần đủ chính xác để collect stats) |
| `-f` | File calibration data |
| `-o`, `--output-file` | Output imatrix file (mặc định `imatrix.gguf`) |
| `-c` | Context length cho mỗi chunk |
| `--chunks` | Số chunk tối đa để xử lý (mặc định -1: toàn bộ file) |
| `-ngl` | Offload lên GPU để thu thập nhanh hơn |

Output: file `imatrix.gguf` (định dạng GGUF là mặc định hiện nay; định dạng `.dat` cũ vẫn dùng được qua `--output-format dat` và có thể chuyển đổi qua lại bằng `--in-file`). Xem thống kê bằng `./build/bin/llama-imatrix --in-file imatrix.gguf --show-statistics`.

---

## Phần 4. Quantize với Imatrix

```bash
# Quantize IQ4_XS với imatrix
./build/bin/llama-quantize \
    --imatrix imatrix.gguf \
    model-f16.gguf \
    model-iq4_xs.gguf \
    IQ4_XS

# K-quant cũng hưởng lợi từ imatrix
./build/bin/llama-quantize --imatrix imatrix.gguf model-f16.gguf model-q4_k_m-imat.gguf Q4_K_M
```

### Các I-quant types phổ biến

| Quant | Bits/param | Kích thước tương đối | Chất lượng |
|:---|:---|:---|:---|
| IQ2_XXS | 2.06 | Rất nhỏ | Thấp (bắt buộc imatrix) |
| IQ2_XS | 2.31 | Rất nhỏ | Thấp-trung bình (bắt buộc imatrix) |
| IQ3_XXS | 3.06 | Nhỏ | Trung bình (bắt buộc imatrix) |
| IQ3_S | 3.44 | Nhỏ | Khá |
| IQ4_XS | 4.25 | Nhỏ-vừa | Tốt |
| IQ4_NL | 4.50 | Vừa | Rất tốt |

Bits/param ở đây là bpw của chính block (ví dụ IQ4_XS: 136 byte cho 256 weight, $136 \times 8 / 256 = 4.25$); file thực tế lớn hơn một chút vì embedding/output thường giữ ở kiểu cao hơn. Danh sách đầy đủ in ra bằng `./build/bin/llama-quantize --help`; upstream hiện còn có các kiểu như `MXFP4`, `NVFP4` và hai kiểu bit thấp `Q1_0`/`Q2_0` dành cho model vốn đã nhị phân/ternary (họ Bonsai, xem [Bài 11](../lesson_11_bonsai_model_family)).

---

## Phần 5. So sánh: Default vs Imatrix Quant

```bash
# Q4_K_M không imatrix
./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Imatrix IQ4_XS
./build/bin/llama-quantize --imatrix imatrix.gguf \
    model-f16.gguf model-iq4_xs.gguf IQ4_XS

# So sánh perplexity trên tập test (khác tập dùng để calibrate)
./build/bin/llama-perplexity -m model-q4_k_m.gguf -f wikitext-2-raw/wiki.test.raw -c 512
./build/bin/llama-perplexity -m model-iq4_xs.gguf -f wikitext-2-raw/wiki.test.raw -c 512
```

### Kết quả kỳ vọng

| Model | PPL (thấp hơn = tốt hơn) | Size |
|:---|:---|:---|
| FP16 (baseline) | 5.82 | 100% |
| Q8_0 (không imatrix) | 5.87 | ~53% |
| Q4_K_M (không imatrix) | 6.25 | ~30% |
| IQ4_XS (imatrix) | 6.10 | ~27% |

Đây là số liệu minh họa xu hướng; giá trị tuyệt đối phụ thuộc model và cấu hình đo. Imatrix thường cải thiện PPL rõ rệt so với quant cùng kích thước không có imatrix, và mức cải thiện tăng dần khi số bit giảm (rõ nhất ở 2 đến 3 bit).

---

## Phần 6. Best Practices

1. **Calibration data chất lượng**: dùng text cùng domain với use case (code data cho code model, Vietnamese data cho Vietnamese model).
2. **Đủ dữ liệu**: vài trăm chunk 512 token (khoảng $10^5$ token trở lên) thường đủ; tăng thêm cho lợi ích giảm dần.
3. **Model đủ chính xác**: dùng FP16 hoặc Q8_0 để collect imatrix, không dùng Q4.
4. **Context length phù hợp**: 512-2048 tokens, tương ứng context length thực tế.

---

## Phần 7. Troubleshooting

| Vấn đề | Nguyên nhân | Fix |
|:---|:---|:---|
| Cảnh báo tensor thiếu dữ liệu imatrix | Calibration quá ngắn, hoặc với MoE một số expert không được kích hoạt | Tăng lượng dữ liệu / đa dạng hóa nội dung |
| PPL không cải thiện | Calibration data khác domain | Dùng data phù hợp hơn |
| OOM khi collect imatrix | Model quá lớn cho RAM | Dùng Q8_0 thay vì FP16 |
| Imatrix không tương thích | Model khác version | Collect lại imatrix cho đúng model |

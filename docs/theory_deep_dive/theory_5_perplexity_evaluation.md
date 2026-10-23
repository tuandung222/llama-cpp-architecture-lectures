---
sidebar_position: 6
sidebar_label: "Bài 5: Perplexity & Evaluation Metrics"
---

# Bài 5: Perplexity & Evaluation Metrics

Làm thế nào để đánh giá chất lượng của một mô hình sau quantization? Bài này trình bày cơ sở toán học của perplexity, cách `llama-perplexity` (`tools/perplexity/perplexity.cpp`) thực sự tính nó, vì sao KL divergence là thước đo tốt hơn cho quantization, và cách kiểm định thống kê đúng đắn khi so sánh hai bản quantize.

---

## 1. Cross-Entropy Loss

Cho sequence $x = (x_1, x_2, ..., x_T)$ và mô hình ngôn ngữ $p_\theta$, negative log-likelihood trung bình (tính bằng nats) là:

$$H = -\frac{1}{T} \sum_{t=1}^{T} \ln p_\theta(x_t | x_{<t})$$

Trong đó $p_\theta(x_t | x_{<t})$ là xác suất mô hình gán cho token thực tế $x_t$ tại vị trí $t$, tính bằng softmax trên logits. Đại lượng này là ước lượng Monte Carlo của cross-entropy $H(p_{\text{data}}, p_\theta)$ giữa phân phối dữ liệu thật và mô hình.

---

## 2. Perplexity Definition

$$PPL = \exp(H) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \ln p_\theta(x_t | x_{<t})\right)$$

**Interpretation**: PPL là "effective branching factor" trung bình. PPL = 10 nghĩa là, tính trung bình theo nghĩa hình học, mô hình "phân vân" như thể chọn đều giữa 10 tokens tại mỗi step.

**PPL thấp hơn = mô hình dự đoán văn bản này tốt hơn**. Lưu ý PPL phụ thuộc tokenizer: hai mô hình với tokenizer khác nhau (ví dụ Llama 2 với 32000 token và Llama 3 với 128256 token) không thể so PPL trực tiếp, vì cùng một văn bản được chia thành số token khác nhau.

---

## 3. Bits-per-Token và Bits-per-Character

Đổi cơ số logarit từ $e$ sang 2:

$$\text{bits/token} = \frac{H}{\ln 2} = \log_2(PPL)$$

Đây là số bit trung bình cần để mã hóa mỗi **token** bằng một bộ mã số học dùng $p_\theta$. Để có đại lượng không phụ thuộc tokenizer, quy đổi sang **bits-per-character** (BPC) hoặc bits-per-byte:

$$BPC = \log_2(PPL) \cdot \frac{N_{\text{tokens}}}{N_{\text{characters}}}$$

Với quantization, điều ta quan tâm là **độ chênh** $\Delta H = H_{Q} - H_{\text{base}}$ (nats/token), và $\ln(PPL_Q / PPL_{\text{base}}) = \Delta H$ chính xác. Một độ tăng PPL tương đối nhỏ xấp xỉ $\Delta H$: ví dụ $\Delta H = 0.01$ nats/token tương ứng PPL tăng khoảng 1%.

---

## 4. llama.cpp Perplexity Computation

```bash
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw -c 512
# giới hạn số chunk để đo nhanh:
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw -c 2048 --chunks 10
```

`llama-perplexity` đặt `n_ctx = 512` làm mặc định riêng (khác các công cụ khác). Methodology trong hàm `perplexity()`:

1. Tokenize toàn bộ file, chia thành `n_chunk` chunk dài `n_ctx` token (không chồng lấn).
2. Với mỗi chunk: xóa memory (`llama_memory_clear(llama_get_memory(ctx), true)`), chạy forward cho cả chunk (chia theo `n_batch`), lấy logits mọi vị trí.
3. **Chỉ chấm điểm nửa sau của chunk**: `const int first = n_ctx/2;`, cộng dồn $-\ln p$ cho các token từ `first` đến cuối chunk (mỗi chunk đóng góp `n_ctx - first - 1` token). Lý do (comment trong mã): token ở đầu chunk gần như không có context, nên nếu chấm điểm chúng, PPL bị thổi phồng và không phản ánh khả năng của mô hình khi có ngữ cảnh.
4. Report: `Final estimate: PPL = ... +/- ...`.

Sai số chuẩn được tính bằng **delta method**. Gọi $\ell_t = -\ln p(x_t)$ với trung bình $\bar{\ell}$ và độ lệch chuẩn mẫu $s_\ell$ trên $N$ token được chấm điểm:

$$PPL = e^{\bar{\ell}}, \qquad SE(\bar{\ell}) = \frac{s_\ell}{\sqrt{N - 1}}, \qquad SE(PPL) \approx \left|\frac{d e^{x}}{dx}\right|_{\bar{\ell}} SE(\bar{\ell}) = PPL \cdot \frac{s_\ell}{\sqrt{N-1}}$$

Đúng như mã: `nll2 = sqrt(nll2/(count-1)); ... PPL = %.4lf +/- %.5lf, ppl, nll2*ppl`. Lưu ý công thức giả định các $\ell_t$ độc lập; các token trong cùng chunk tương quan với nhau nên sai số thật lớn hơn một chút.

Vì methodology này (context 512, chỉ nửa sau, không chồng lấn), con số của `llama-perplexity` **không so được trực tiếp** với PPL trong các paper (thường đo ở context 2048 với sliding window khác). Chỉ so các mô hình đo bằng cùng công cụ, cùng file, cùng `-c`.

---

## 5. KL Divergence: thước đo đúng cho quantization

PPL chỉ nhìn xác suất của **token đúng**. Hai mô hình có thể có PPL gần nhau nhưng phân phối trên các token khác rất khác. Với quantization, câu hỏi thực sự là "mô hình quantized lệch khỏi mô hình gốc bao nhiêu", và thước đo tự nhiên là KL divergence giữa hai phân phối dự đoán tại mỗi vị trí:

$$D_{KL}(p_{\text{base}} \,\|\, p_Q)_t = \sum_{v=1}^{|V|} p_{\text{base}}(v \mid x_{<t}) \ln \frac{p_{\text{base}}(v \mid x_{<t})}{p_Q(v \mid x_{<t})}$$

Quy trình hai bước của `llama-perplexity`:

```bash
# 1. Chạy mô hình gốc, lưu log-prob của mọi token mọi vị trí (dạng nén uint16)
./build/bin/llama-perplexity -m model-f16.gguf -f wiki.test.raw --kl-divergence-base logits-f16.bin
# 2. Chạy mô hình quantized, so với file base
./build/bin/llama-perplexity -m model-q4_k_m.gguf --kl-divergence-base logits-f16.bin --kl-divergence
```

Bước 2 in ba nhóm thống kê (hàm `kl_divergence()`):

- **Perplexity statistics**: `Mean PPL(Q)`, `Mean PPL(base)`, `Cor(ln(PPL(Q)), ln(PPL(base)))`, `Mean ln(PPL(Q)/PPL(base))`, `Mean PPL(Q)/PPL(base)`, `Mean PPL(Q)-PPL(base)`, mỗi giá trị kèm sai số.
- **KL divergence statistics**: `Mean KLD`, cùng phân vị (`Maximum`, `99.9%`, `99.0%`, ..., `Median`, ..., `Minimum`). Đuôi phân phối (99.9%) rất quan trọng: một bản quantize có KLD trung bình nhỏ nhưng thỉnh thoảng lệch rất mạnh sẽ gây lỗi "đột ngột" khi sinh văn bản dài.
- **Token probability statistics**: `Mean Δp` (độ thay đổi xác suất của token đúng), `RMS Δp`, và `Same top p` (tỷ lệ vị trí mà token có xác suất cao nhất trùng nhau giữa hai mô hình; với greedy decoding, đây gần như là tỷ lệ token đầu ra giống hệt).

---

## 6. Statistical Significance Testing

Khi so sánh PPL giữa hai quant levels, một sai lầm phổ biến là dùng t-test hai mẫu độc lập:

$$t = \frac{\overline{\ell_1} - \overline{\ell_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Công thức này sai cho bài toán của chúng ta, vì hai mô hình được đánh giá trên **cùng** các token: $\ell_{1,t}$ và $\ell_{2,t}$ tương quan rất mạnh (một token khó thì khó với cả hai). Cách đúng là kiểm định **ghép cặp** (paired) trên hiệu $d_t = \ell_{Q,t} - \ell_{\text{base},t}$:

$$\bar{d} = \ln\frac{PPL_Q}{PPL_{\text{base}}}, \qquad \text{Var}(\bar{d}) = \text{Var}(\bar{\ell}_Q) + \text{Var}(\bar{\ell}_{\text{base}}) - 2\,\text{Cov}(\bar{\ell}_Q, \bar{\ell}_{\text{base}})$$

Đây chính xác là công thức `log_ppl_ratio_unc = sqrt(var_Q + var_base - 2*cov)` trong `perplexity.cpp`. Khi tương quan (dòng `Cor(...)`) lên tới 99% như thường thấy, số hạng hiệp phương sai triệt tiêu gần hết phương sai, nên sai số của **tỷ lệ** PPL nhỏ hơn nhiều so với sai số của từng PPL riêng lẻ. Hệ quả thực tế: hai bản quantize có khoảng tin cậy PPL chồng lấn nhau vẫn có thể khác nhau có ý nghĩa thống kê theo kiểm định ghép cặp. Với $|\bar{d}| / SE(\bar{d}) > 1.96$, sự khác biệt có ý nghĩa ở mức $p < 0.05$ (xấp xỉ chuẩn, cùng giả định độc lập giữa các vị trí như ở Phần 4).

---

## 7. Benchmark Methodology

| Benchmark | Metric | Mô tả | Chạy trong llama.cpp |
|:---|:---|:---|:---|
| WikiText-2 | PPL, KLD | General language modeling | `llama-perplexity -f wiki.test.raw` |
| HellaSwag | Accuracy | Commonsense reasoning (chọn kết thúc hợp lý) | `llama-perplexity --hellaswag` |
| WinoGrande | Accuracy | Giải quyết đồng tham chiếu | `llama-perplexity --winogrande` |
| MMLU, ARC, TruthfulQA (dạng trắc nghiệm) | Accuracy | Multi-task knowledge | `llama-perplexity --multiple-choice` với file dữ liệu phù hợp |
| LAMBADA | Accuracy | Zero-shot completion từ cuối | Công cụ ngoài (ví dụ lm-evaluation-harness qua `llama-server`) |
| HumanEval | pass@1 | Code generation | Công cụ ngoài, cần sinh văn bản và chạy test |

Các chế độ `--hellaswag`, `--winogrande`, `--multiple-choice` chấm điểm bằng log-likelihood của từng phương án (không sinh văn bản), nên nhanh và tất định. Benchmark sinh văn bản như HumanEval nhạy với sampling và chat template, cần cố định seed và cấu hình sampler.

---

## 💡 Đúc kết

Perplexity là metric cơ bản, nhưng để đánh giá quantization, hãy ưu tiên **KL divergence so với mô hình gốc** (đặc biệt phân vị 99%/99.9%) và `Same top p`, vì chúng đo trực tiếp mức lệch phân phối. Khi so sánh, dùng kiểm định ghép cặp như `llama-perplexity` đã làm sẵn, và chỉ so các con số đo bằng cùng công cụ, cùng dữ liệu, cùng độ dài context. Ở Llama-3-8B, bảng tham chiếu trong `tools/quantize/quantize.cpp` cho thấy Q4_K_M tăng PPL khoảng +0.18 (cỡ 3% nếu PPL gốc khoảng 6), một trade-off chấp nhận được; kết hợp PPL/KLD với task-specific benchmarks cho bức tranh toàn diện.

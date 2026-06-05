---
sidebar_position: 6
sidebar_label: "Bài 5: Perplexity & Evaluation Metrics"
---

# Bài 5: Perplexity & Evaluation Metrics

Làm thế nào để đánh giá chất lượng của một mô hình sau quantization? Bài này trình bày cơ sở toán học của perplexity, methodology benchmarking, và statistical testing.

---

## 1. Cross-Entropy Loss

Cho sequence $x = (x_1, x_2, ..., x_T)$ và mô hình ngôn ngữ $p_\theta$:

$$H(p, q) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

Trong đó $p_\theta(x_t | x_{<t})$ là xác suất mô hình gán cho token thực tế $x_t$ tại vị trí $t$.

---

## 2. Perplexity Definition

$$PPL = \exp(H(p, q)) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})\right)$$

**Interpretation**: PPL là "effective branching factor" trung bình. PPL = 10 nghĩa là mô hình "phân vân" giữa ~10 tokens tại mỗi step.

**PPL thấp hơn = mô hình tốt hơn**.

---

## 3. Bits-per-Character và Bits-per-Word

$$BPC = \frac{H}{\ln 2} = \log_2(PPL)$$

BPC cho biết số bits cần thiết để encode mỗi token. Quantization lý thuyết: nếu trọng số bị nén $b$ bits/weight, BPC tăng tối đa $\sim \frac{\Delta b}{d}$ bits/token.

---

## 4. llama.cpp Perplexity Computation

```bash
./llama-perplexity -m model.gguf -f wikitext-2.txt -c 2048 --chunks 10
```

Methodology:
1. Chia text thành chunks có độ dài `n_ctx`.
2. Với mỗi chunk, tính log probability cho tất cả tokens.
3. Trung bình cross-entropy trên tất cả chunks.
4. Report: PPL ± standard error.

---

## 5. Benchmark Methodology

| Benchmark | Metric | Mô tả |
|:---|:---|:---|
| WikiText-2 | PPL | General language modeling |
| LAMBADA | Accuracy | Zero-shot completion |
| MMLU | Accuracy | Multi-task knowledge |
| HellaSwag | Accuracy | Commonsense reasoning |
| HumanEval | pass@1 | Code generation |

---

## 6. Statistical Significance Testing

Khi so sánh PPL giữa hai quant levels:

$$t = \frac{\overline{PPL_1} - \overline{PPL_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Với $p < 0.05$, sự khác biệt có ý nghĩa thống kê.

---

## 💡 Đúc kết

Perplexity là metric chính cho đánh giá chất lượng quantization. Q4_K_M thường tăng PPL ~3% so với FP16, đây là trade-off chấp nhận được. Kết hợp PPL với task-specific benchmarks (MMLU, HumanEval) cho bức tranh toàn diện.

---
sidebar_position: 6
sidebar_label: "Case 5: Chuyển đổi LoRA Adapter sang GGUF"
---

# Case 5: LoRA Adapter Merging và Export sang GGUF

## 1. Bối cảnh

LoRA (Low-Rank Adaptation) cho phép fine-tune LLM với chi phí thấp bằng cách chỉ huấn luyện một ma trận nhỏ $\Delta W = BA$ thay vì toàn bộ trọng số. Tuy nhiên, LoRA adapter thường ở dạng PyTorch (.bin/.safetensors) và cần được convert sang GGUF để dùng với llama.cpp.

Case study này đi qua toàn bộ quy trình: từ LoRA adapter -> merge vào base model -> export GGUF -> quantize.

---

## 2. Kiến thức nền: LoRA Math

LoRA thay thế weight update $\Delta W$ bằng tích hai ma trận hạng thấp:

$$W' = W + \Delta W = W + BA$$

Trong đó:
- $W \in \mathbb{R}^{d \times k}$: weight gốc (đóng băng).
- $B \in \mathbb{R}^{d \times r}$: ma trận down-projection.
- $A \in \mathbb{R}^{r \times k}$: ma trận up-projection.
- $r \ll \min(d, k)$: rank, thường 8, 16, 32, 64.

Khi $r = 16$, số tham số LoRA chỉ bằng $\frac{r(d+k)}{d \times k} \approx 0.5\%$ so với full fine-tune.

---

## 3. Quy trình end-to-end

### Bước 1: Huấn luyện LoRA (bằng PEFT/Axolotl)

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)
# ... huấn luyện bình thường ...
model.save_pretrained("lora-adapter/")
```

Output: thư mục `lora-adapter/` chứa `adapter_model.safetensors` và `adapter_config.json`.

### Bước 2: Merge LoRA vào Base Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype="auto",
)

# Load và merge LoRA
model = PeftModel.from_pretrained(base, "lora-adapter/")
merged = model.merge_and_unload()

# Save merged model
merged.save_pretrained("merged-model/", safe_serialization=True)
```

Sau bước này, `merged-model/` chứa full weights đã gộp LoRA.

### Bước 3: Convert sang GGUF

```bash
# Convert merged model sang GGUF FP16
python convert_hf_to_gguf.py merged-model/ \
    --outfile merged-f16.gguf \
    --outtype f16

# Quantize xuống Q4_K_M
./llama-quantize merged-f16.gguf merged-q4_k_m.gguf Q4_K_M
```

---

## 4. Dùng LoRA trực tiếp trong llama.cpp (không merge)

llama.cpp cũng hỗ trợ load LoRA adapter riêng biệt mà không cần merge trước:

```bash
./llama-cli \
    -m base-model-q4_k_m.gguf \
    --lora adapter-f16.gguf \
    -p "..." \
    -n 200
```

### Convert adapter riêng sang GGUF

```bash
python convert_lora_to_gguf.py lora-adapter/ \
    --base merged-model/ \
    --outfile adapter-f16.gguf
```

Ưu điểm của cách này:
- **Tiết kiệm bộ nhớ**: base model quant Q4 + adapter FP16 nhẹ hơn nhiều so với full merge.
- **Linh hoạt**: swap nhiều adapter khác nhau trên cùng base model.
- **Nhanh**: không cần merge lại mỗi khi đổi adapter.

### Scaling alpha

llama.cpp tự động tính scaling factor:

$$\text{scale} = \frac{\alpha}{r}$$

Với $\alpha = 32, r = 16$: scale = 2.0. Hệ số này nhân vào $\Delta W$ khi apply:

$$W'_{applied} = W + \frac{\alpha}{r} \cdot BA$$

---

## 5. Multi-LoRA Stacking

llama.cpp cho phép load nhiều LoRA cùng lúc:

```bash
./llama-cli \
    -m base.gguf \
    --lora adapter-code-f16.gguf \
    --lora adapter-vietnamese-f16.gguf \
    --lora-scale 0.7 0.3 \
    -p "..."
```

Flag `--lora-scale` kiểm soát trọng số từng adapter. Ví dụ trên blend 70% code capability + 30% Vietnamese capability.

---

## 6. Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|:---|:---|:---|
| Chất lượng kém sau merge | alpha/r ratio sai | Kiểm tra `lora_alpha` trong config |
| OOM khi merge full model | RAM không đủ cho FP16 | Dùng `--outtype q8_0` thay vì f16 |
| Logits mismatch | Target modules thiếu | Đảm bảo LoRA cover tất cả linear layers |
| Shape mismatch khi load | Vocab size thay đổi | Không thêm special tokens vào tokenizer |

## 7. Kết luận

LoRA + GGUF là combo mạnh cho production: fine-tune rẻ bằng LoRA, deploy hiệu quả bằng GGUF quantization. llama.cpp hỗ trợ cả hai workflow (merge trước hoặc load adapter riêng), cho phép linh hoạt tối đa trong việc quản lý và phục vụ nhiều model fine-tuned trên cùng một base model.

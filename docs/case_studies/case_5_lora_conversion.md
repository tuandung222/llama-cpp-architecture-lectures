---
sidebar_position: 6
sidebar_label: "Case 5: Chuyển đổi LoRA Adapter sang GGUF"
---

# Case 5: LoRA Adapter Merging và Export sang GGUF

## Phần 1. Bối cảnh

LoRA (Low-Rank Adaptation) cho phép fine-tune LLM với chi phí thấp bằng cách chỉ huấn luyện một ma trận nhỏ $\Delta W = BA$ thay vì toàn bộ trọng số. Tuy nhiên, LoRA adapter thường ở dạng PyTorch (.bin/.safetensors) và cần được convert sang GGUF để dùng với llama.cpp.

Case study này đi qua toàn bộ quy trình: từ LoRA adapter -> merge vào base model -> export GGUF -> quantize.

---

## Phần 2. Kiến thức nền: LoRA Math

LoRA thay thế weight update $\Delta W$ bằng tích hai ma trận hạng thấp:

$$W' = W + \Delta W = W + BA$$

Trong đó:
- $W \in \mathbb{R}^{d \times k}$: weight gốc (đóng băng).
- $A \in \mathbb{R}^{r \times k}$: ma trận **down-projection** (chiếu input $x \in \mathbb{R}^k$ xuống không gian hạng $r$), khởi tạo ngẫu nhiên.
- $B \in \mathbb{R}^{d \times r}$: ma trận **up-projection** (chiếu ngược lên $\mathbb{R}^d$), khởi tạo bằng 0 để $\Delta W = 0$ lúc bắt đầu.
- $r \ll \min(d, k)$: rank, thường 8, 16, 32, 64.

Tỷ lệ tham số LoRA trên một ma trận là $\frac{r(d+k)}{d \times k}$. Với ma trận vuông $d = k = 4096$ và $r = 16$, tỷ lệ này bằng $\frac{2r}{d} = \frac{32}{4096} \approx 0.78\%$; với các ma trận chữ nhật của FFN (ví dụ $d = 14336$, $k = 4096$) tỷ lệ còn nhỏ hơn, khoảng $0.5\%$.

---

## Phần 3. Quy trình end-to-end

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
# Convert merged model sang GGUF (bf16 giữ nguyên dtype của Llama 3.2)
python convert_hf_to_gguf.py merged-model/ \
    --outfile merged-bf16.gguf \
    --outtype bf16

# Quantize xuống Q4_K_M
./build/bin/llama-quantize merged-bf16.gguf merged-q4_k_m.gguf Q4_K_M
```

Lưu ý: `merged.save_pretrained()` chỉ lưu weights và config; hãy lưu thêm tokenizer (`tokenizer.save_pretrained("merged-model/")`) để converter đọc được vocab.

---

## Phần 4. Dùng LoRA trực tiếp trong llama.cpp (không merge)

llama.cpp cũng hỗ trợ load LoRA adapter riêng biệt mà không cần merge trước:

```bash
./build/bin/llama-cli \
    -m base-model-q4_k_m.gguf \
    --lora adapter-f16.gguf \
    -p "..." \
    -n 200 -st
```

### Convert adapter riêng sang GGUF

```bash
python convert_lora_to_gguf.py lora-adapter/ \
    --base Llama-3.2-3B/ \
    --outtype f16 \
    --outfile adapter-f16.gguf
```

`--base` trỏ tới thư mục của **base model gốc** (chỉ cần `config.json` và tokenizer, không cần weights), không phải merged model; nếu bỏ qua, script đọc `base_model_name_or_path` từ `adapter_config.json` và tải config từ Hugging Face (hoặc chỉ định `--base-model-id`). Mặc định `--outtype` là `f32`, nên chỉ rõ `f16` hoặc `bf16` để adapter nhỏ hơn.

Ưu điểm của cách này:
- **Tiết kiệm bộ nhớ**: base model quant Q4 + adapter FP16 nhẹ hơn nhiều so với full merge.
- **Linh hoạt**: swap nhiều adapter khác nhau trên cùng base model.
- **Nhanh**: không cần merge lại mỗi khi đổi adapter.

### Scaling alpha

`convert_lora_to_gguf.py` ghi `lora_alpha` vào metadata `adapter.lora.alpha` của GGUF; khi nạp adapter (`src/llama-adapter.cpp`), llama.cpp đọc giá trị này và tính hệ số:

$$\text{scale}_{\text{eff}} = s_{\text{user}} \cdot \frac{\alpha}{r}$$

với $s_{\text{user}}$ là hệ số người dùng truyền vào (mặc định 1.0 khi dùng `--lora`). Với $\alpha = 32, r = 16$ và $s_{\text{user}} = 1$: hệ số bằng 2.0. Adapter được áp dụng ở runtime, không sửa weight gốc:

$$y = W x + s_{\text{eff}} \cdot B (A x)$$

Bản chất của công thức là llama.cpp tính nhánh LoRA như hai phép nhân ma trận nhỏ ($A x$ rồi $B(\cdot)$) cộng vào output của base, thay vì tạo ra $W'$ đầy đủ.

---

## Phần 5. Multi-LoRA Stacking

llama.cpp cho phép load nhiều LoRA cùng lúc:

```bash
./build/bin/llama-cli \
    -m base.gguf \
    --lora-scaled adapter-code-f16.gguf:0.7,adapter-vietnamese-f16.gguf:0.3 \
    -p "..." -st
```

Flag `--lora-scaled FNAME:SCALE,...` đặt $s_{\text{user}}$ riêng cho từng adapter; `--lora` nhận danh sách file phân cách bằng dấu phẩy với hệ số 1.0. Ví dụ trên cộng $0.7 \cdot \Delta W_{\text{code}} + 0.3 \cdot \Delta W_{\text{vi}}$ (mỗi hạng tử đã nhân $\alpha/r$ của adapter đó). Đây là phép cộng tuyến tính các update, không bảo đảm "70% năng lực code + 30% năng lực tiếng Việt"; cần đánh giá thực nghiệm. Với `llama-server`, có thể nạp adapter bằng `--lora-init-without-apply` rồi đổi hệ số theo từng request qua endpoint `/lora-adapters`.

---

## Phần 6. Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|:---|:---|:---|
| Chất lượng kém sau merge | alpha/r ratio sai | Kiểm tra `lora_alpha` trong config |
| OOM khi merge full model | RAM không đủ cho FP16 | Merge trên máy nhiều RAM hơn, hoặc dùng adapter riêng (Phần 4); `--outtype q8_0` của converter chỉ giảm kích thước file đầu ra |
| Logits mismatch | Target modules thiếu | Đảm bảo LoRA cover tất cả linear layers |
| Shape mismatch khi load | Vocab size thay đổi | Không thêm special tokens vào tokenizer, hoặc merge rồi convert lại toàn bộ model |
| Adapter không có tác dụng | Adapter convert với base khác | Dùng đúng base model (cùng kiến trúc và kích thước) khi convert và khi chạy |

## Phần 7. Kết luận

LoRA + GGUF là combo mạnh cho production: fine-tune rẻ bằng LoRA, deploy hiệu quả bằng GGUF quantization. llama.cpp hỗ trợ cả hai workflow (merge trước hoặc load adapter riêng), cho phép linh hoạt tối đa trong việc quản lý và phục vụ nhiều model fine-tuned trên cùng một base model.

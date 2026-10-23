---
sidebar_position: 3
sidebar_label: "Case 2: Porting Llama 3 từ HuggingFace sang GGUF"
---

# Case 2: Porting Llama 3 từ HuggingFace sang GGUF

## Phần 1. Bối cảnh

Llama 3 khác Llama 2 ở một số điểm: GQA (Grouped Query Attention) ở mọi kích thước, tokenizer BPE mới kiểu tiktoken với vocab 128256, RoPE base 500000. Case study này đi qua quy trình convert end-to-end bằng llama.cpp hiện tại.

## Phần 2. Step-by-step Conversion

```bash
# 0. Môi trường Python cho converter (trong thư mục llama.cpp)
pip install -r requirements.txt

# 1. Tải model từ Hugging Face (model gated: cần chấp nhận license và đăng nhập)
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir Meta-Llama-3-8B

# 2. Convert sang GGUF BF16 (giữ nguyên dtype gốc, tránh tràn số của FP16)
python convert_hf_to_gguf.py Meta-Llama-3-8B/ \
    --outfile llama-3-8b-bf16.gguf \
    --outtype bf16

# 3. Quantize xuống Q4_K_M
./build/bin/llama-quantize llama-3-8b-bf16.gguf llama-3-8b-q4_k_m.gguf Q4_K_M

# 4. Test (model base: dùng llama-completion ở chế độ không hội thoại)
./build/bin/llama-completion -m llama-3-8b-q4_k_m.gguf \
    -p "Explain quantum computing:" -n 100 -no-cnv
```

Với bản **Instruct**, chat template được đọc từ metadata `tokenizer.chat_template` trong GGUF, nên `llama-cli` dùng đúng template mà không cần cờ thêm; `--chat-template llama3` chỉ cần khi metadata thiếu hoặc sai:

```bash
./build/bin/llama-cli -m llama-3-8b-instruct-q4_k_m.gguf -p "Explain quantum computing" -n 200 -st
```

(`-st`/`--single-turn` chạy một lượt rồi thoát; bỏ cờ này để vào chế độ chat tương tác.)

## Phần 3. Weight Mapping Chi tiết

Converter **không transpose** các ma trận linear: một tensor PyTorch shape `[out, in]` (row-major) có cùng bố cục bộ nhớ với tensor GGML `ne = [in, out]`, vì GGML ghi chiều chạy nhanh nhất trước. Biến đổi đặc biệt duy nhất của Llama là **hoán vị lại hàng của Q và K** (`LlamaModel.permute()` trong `conversion/llama.py`) để đưa layout RoPE kiểu Hugging Face về layout mà kernel RoPE của GGML mong đợi.

| PyTorch Layer | GGUF Name | PyTorch shape `[out, in]` | GGUF `ne` `[ne0, ne1]` | Transform |
|:---|:---|:---|:---|:---|
| model.embed_tokens | token_embd | [128256, 4096] | [4096, 128256] | None |
| layers.N.self_attn.q_proj | blk.N.attn_q | [4096, 4096] | [4096, 4096] | Permute (RoPE) |
| layers.N.self_attn.k_proj | blk.N.attn_k | [1024, 4096] | [4096, 1024] | Permute (RoPE, GQA) |
| layers.N.self_attn.v_proj | blk.N.attn_v | [1024, 4096] | [4096, 1024] | None |
| layers.N.self_attn.o_proj | blk.N.attn_output | [4096, 4096] | [4096, 4096] | None |
| layers.N.mlp.gate_proj | blk.N.ffn_gate | [14336, 4096] | [4096, 14336] | None |
| layers.N.mlp.up_proj | blk.N.ffn_up | [14336, 4096] | [4096, 14336] | None |
| layers.N.mlp.down_proj | blk.N.ffn_down | [4096, 14336] | [14336, 4096] | None |
| lm_head | output | [128256, 4096] | [4096, 128256] | None |

Có thể tự kiểm tra bằng `python gguf-py/gguf/scripts/gguf_dump.py llama-3-8b-bf16.gguf`, công cụ in tên, `ne` và kiểu của từng tensor.

## Phần 4. Validation

Upstream cung cấp sẵn quy trình so sánh logits giữa model Hugging Face gốc và model GGUF trong `examples/model-conversion`:

```bash
cd examples/model-conversion
pip install -r requirements.txt
export MODEL_PATH=$(realpath ../../Meta-Llama-3-8B)
make causal-convert-model           # convert sang GGUF, in ra đường dẫn file .gguf
export CONVERTED_MODEL=<đường dẫn .gguf vừa in ra>
make causal-verify-logits           # chạy PyTorch và llama.cpp (llama-debug), so sánh logits và NMSE
```

Về mặt toán học, phép so sánh cốt lõi là độ tương đồng cosine giữa hai vector logits $\ell_{\text{pt}}, \ell_{\text{gguf}} \in \mathbb{R}^{128256}$ của cùng một vị trí token:

```python
import numpy as np
cosine_sim = np.dot(pt_logits, gguf_logits) / (np.linalg.norm(pt_logits) * np.linalg.norm(gguf_logits))
assert cosine_sim > 0.999, f"Logits mismatch: cosine_sim = {cosine_sim}"
```

Ngưỡng 0.999 áp dụng cho so sánh BF16/F16 với bản gốc; sau khi quantize, nên đánh giá bằng PPL và KL divergence (Case 1).

## Phần 5. Common Pitfalls

- **GQA**: `k_proj` và `v_proj` có số hàng bằng $n_{kv\_heads} \times d_{head} = 8 \times 128 = 1024$, khác với 32 heads của `q_proj`.
- **Tokenizer**: Llama 3 dùng BPE kiểu tiktoken, không phải SentencePiece. Converter nhận diện pre-tokenizer bằng hash của tokenizer (`get_vocab_base_pre()` trong `conversion/base.py`) và ghi `tokenizer.ggml.pre = "llama-bpe"`; nếu gặp cảnh báo pre-tokenizer không nhận diện được, tokenizer đã bị sửa so với bản gốc.
- **RoPE base**: 500000 thay vì 10000, được ghi tự động vào `llama.rope.freq_base` từ `rope_theta` của `config.json`.
- **Tied embeddings**: Llama 3 8B và 70B **không** tied (có `lm_head` riêng); Llama 3.2 1B/3B thì tied, khi đó GGUF không có tensor `output` và llama.cpp dùng lại `token_embd`.

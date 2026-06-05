---
sidebar_position: 3
sidebar_label: "Case 2: Porting Llama 3 từ HuggingFace sang GGUF"
---

# Case 2: Porting Llama 3 từ HuggingFace sang GGUF

## 1. Bối cảnh

Llama 3 có kiến trúc khác Llama 2 ở một số điểm: GQA (Grouped Query Attention), tokenizer mới (tiktoken-based), RoPE base 500000. Case study này đi qua quy trình convert end-to-end.

## 2. Step-by-step Conversion

```bash
# 1. Clone model từ HuggingFace
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B

# 2. Convert sang GGUF FP16
python convert_hf_to_gguf.py Meta-Llama-3-8B/ \
    --outfile llama-3-8b-f16.gguf \
    --outtype f16

# 3. Quantize xuống Q4_K_M
./llama-quantize llama-3-8b-f16.gguf llama-3-8b-q4_k_m.gguf Q4_K_M

# 4. Test
./llama-cli -m llama-3-8b-q4_k_m.gguf \
    -p "Explain quantum computing:" \
    -n 100 --chat-template llama3
```

## 3. Weight Mapping Chi tiết

| PyTorch Layer | GGUF Name | Shape | Transform |
|:---|:---|:---|:---|
| model.embed_tokens | token_embd | [128256, 4096] | None |
| layers.*.self_attn.q_proj | blk.*.attn_q | [4096, 4096] | Transpose |
| layers.*.self_attn.k_proj | blk.*.attn_k | [4096, 1024] | Transpose (GQA) |
| layers.*.self_attn.v_proj | blk.*.attn_v | [4096, 1024] | Transpose (GQA) |
| layers.*.mlp.gate_proj | blk.*.ffn_gate | [4096, 14336] | Transpose |
| lm_head | output | [128256, 4096] | Transpose |

## 4. Validation

So sánh logits giữa PyTorch và llama.cpp cho cùng input:

```python
# PyTorch logits
pt_logits = model(input_ids).logits[0, -1, :].numpy()

# llama.cpp logits (qua Python bindings)
llama_logits = llm.get_logits(input_ids)

# Compare
cosine_sim = np.dot(pt_logits, llama_logits) / (np.linalg.norm(pt_logits) * np.linalg.norm(llama_logits))
assert cosine_sim > 0.999, f"Logits mismatch: cosine_sim = {cosine_sim}"
```

## 5. Common Pitfalls

- **GQA**: k_proj và v_proj có shape khác q_proj (8 heads vs 32 heads).
- **Tokenizer**: Llama 3 dùng tiktoken, không phải SentencePiece.
- **RoPE base**: 500000 thay vì 10000, cần set đúng trong metadata.
- **Tied embeddings**: Llama 3 có thể dùng tied hoặc untied embeddings.

---
sidebar_position: 1
sidebar_label: "Lộ trình Case Studies"
---

# Lộ trình Case Studies Thực chiến với llama.cpp

Chào mừng bạn đến với chương **Case Studies Thực chiến**. Sau khi đã nắm vững lý thuyết cốt lõi (từ Bài 0 đến Bài 9), phần này sẽ đưa bạn vào các kịch bản thực tế khi deploy và tối ưu llama.cpp.

---

```mermaid
graph TD
    CS1[Case 1: Q4_K_M vs Q5_K_M vs Q8_0] --> CS2[Case 2: Porting Llama 3]
    CS2 --> CS3[Case 3: Edge Deployment]
    CS3 --> CS4[Case 4: Speculative Decoding]
    CS4 --> CS5[Case 5: LoRA Conversion]
    CS5 --> CS6[Case 6: bitnet.cpp vs llama.cpp]

    style CS1 fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff
    style CS2 fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff
    style CS3 fill:#92400e,stroke:#f59e0b,stroke-width:2px,color:#fff
    style CS4 fill:#9d174d,stroke:#ec4899,stroke-width:2px,color:#fff
    style CS5 fill:#5b21b6,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style CS6 fill:#059669,stroke:#10b981,stroke-width:2px,color:#fff
```

---

## Nội dung Case Studies

1. **[Case 1: Q4_K_M vs Q5_K_M vs Q8_0](case_1_quant_comparison)** - So sánh chất lượng, tốc độ và bộ nhớ giữa các quant types phổ biến.
2. **[Case 2: Porting Llama 3 từ HuggingFace sang GGUF](case_2_llama3_porting)** - Hướng dẫn chi tiết convert model mới nhất.
3. **[Case 3: Edge Deployment trên Raspberry Pi và Điện thoại](case_3_edge_deployment)** - Tối ưu cho thiết bị giới hạn.
4. **[Case 4: Speculative Decoding - Tăng tốc 2-3x](case_4_speculative_decoding)** - Draft model + target model strategy.
5. **[Case 5: Chuyển đổi LoRA Adapter sang GGUF](case_5_lora_conversion)** - Fine-tuned model conversion pipeline.
6. **[Case 6: bitnet.cpp vs llama.cpp cho Ternary LLM](case_6_bitnet_vs_llamacpp)** - Benchmark performance, accuracy, energy khi chạy ternary model trên cả hai framework.

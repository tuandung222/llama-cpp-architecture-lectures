---
sidebar_position: 8
sidebar_label: "Case 7: Chạy Bonsai 2 trên llama.cpp mainline và fork Prism"
---

# Case 7: Chạy Ternary Bonsai 2 9B trên llama.cpp mainline và fork Prism

## Phần 1: Bối cảnh và mục tiêu

Một nhóm cần một trợ lý đa phương thức chạy cục bộ trên GPU 12 GB hoặc laptop 16 GB, context dài, có tool calling. Ternary Bonsai 2 9B (lý thuyết ở [Bài 11](../lesson_11_bonsai_model_family)) là ứng viên: phần ngôn ngữ chỉ 2.0 đến 2.56 GB tùy gói. Case study này đi qua toàn bộ quy trình:

1. Chọn runtime (mainline hay fork) và gói (`Q2_0`, `PQ2_0`, `PTQ1_0`).
2. Kiểm tra metadata Hadamard trước khi chạy.
3. Lập ngân sách bộ nhớ cho mô hình hybrid attention.
4. Đo throughput và bật speculative decoding.
5. Chẩn đoán các lỗi hay gặp.

## Phần 2: Chọn runtime và gói

| Gói | Kích thước | Runtime chạy được | Khi nào chọn |
|:---|:---:|:---|:---|
| `Ternary-Bonsai-2-9B-Q2_0.gguf` | 2.56 GB | Mainline (bản có runtime Hadamard) và fork | Muốn dùng llama.cpp chính thức |
| `Ternary-Bonsai-2-9B-PQ2_0.gguf` | 2.42 GB | Chỉ fork | Mặc định của PrismML; nhanh trên Metal và GPU datacenter |
| `Ternary-Bonsai-2-9B-PTQ1_0.gguf` | 2.00 GB | Chỉ fork | Bộ nhớ hạn chế; nhanh nhất trên CUDA theo whitepaper Bonsai 2 Small |
| `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` | 2.68 GB | Chỉ fork | Dùng **thay cho** file `PQ2_0` khi muốn speculative decoding MTP |
| `Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf` | 0.62 GB | Cả hai | Khi cần đầu vào ảnh |

```bash
# Tải bằng Hugging Face CLI
hf download prism-ml/Ternary-Bonsai-2-9B-gguf \
  Ternary-Bonsai-2-9B-Q2_0.gguf Ternary-Bonsai-2-9B-PQ2_0.gguf \
  Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf --local-dir models/
```

Build hai runtime cạnh nhau, **không** trộn thư viện của chúng:

```bash
git clone https://github.com/ggml-org/llama.cpp mainline && cmake -S mainline -B mainline/build -DGGML_CUDA=ON && cmake --build mainline/build -j
git clone https://github.com/PrismML-Eng/llama.cpp prism   && cmake -S prism    -B prism/build    -DGGML_CUDA=ON && cmake --build prism/build -j
# macOS: bỏ -DGGML_CUDA=ON, Metal được bật mặc định
```

## Phần 3: Kiểm tra hợp đồng Hadamard trước khi chạy

Một file Bonsai 2 chạy trên runtime không hiểu `prism.hadamard.*` sẽ sinh rác mà không báo lỗi ([Theory 10](../theory_deep_dive/theory_10_hadamard_folding), Phần 6). Vì vậy bước đầu tiên là đọc metadata:

```bash
pip install gguf            # hoặc: cd mainline/gguf-py && pip install -e .
gguf-dump --no-tensors models/Ternary-Bonsai-2-9B-Q2_0.gguf | grep -E "prism\.hadamard|general.architecture|block_count"
```

Cần thấy kiến trúc `qwen35`, `prism.hadamard.version`, `prism.hadamard.block_size` (4096 với bản 9B theo whitepaper), `prism.hadamard.sign_mode` và danh sách `prism.hadamard.weight_names`. Khi nạp bằng runtime, lỗi kiểu `unsupported prism.hadamard...` hay `prism.hadamard: arch ... is not verified` nghĩa là loader đã chủ động từ chối, đúng hành vi mong muốn. Ngược lại, nếu runtime không in gì về Hadamard và văn bản đầu ra vô nghĩa, runtime đó quá cũ.

Phép thử nhanh để phân biệt "runtime đúng" với "runtime thiếu Hadamard": cùng một prompt, bật greedy (`--temp 0`), so đầu ra của mainline với file `Q2_0` và của fork với file `PQ2_0`. Hai gói chứa cùng trọng số ternary (Theory 9, Phần 4), nên đầu ra phải gần như trùng nhau; khác biệt chỉ đến từ thứ tự cộng dồn trong kernel.

## Phần 4: Ngân sách bộ nhớ của mô hình hybrid

Cấu hình phần ngôn ngữ của Qwen3.5-9B (từ `config.json` của base model): 32 tầng, trong đó 8 tầng full attention (`full_attention_interval = 4`) và 24 tầng Gated DeltaNet; full attention có 4 KV head, `head_dim = 256`; GDN có 16 key head, 32 value head, mỗi head 128 chiều; kernel conv 4.

**KV Cache** chỉ tồn tại ở 8 tầng full attention. Với $n_{\text{kv}} = 4$, $d_h = 256$, mỗi token tốn:

$$M_{\text{KV}}/\text{token} = n_{\text{layer}}^{\text{full}} \times 2 \times n_{\text{kv}} \times d_h \times \text{byte} = 8 \times 2 \times 4 \times 256 \times 2 = 32{,}768 \text{ byte (f16)}$$

| Context | f16 | q8_0 (8.5 bit) | q4_0 (4.5 bit) |
|:---:|:---:|:---:|:---:|
| 32,768 | 1.0 GiB | 0.53 GiB | 0.28 GiB |
| 131,072 | 4.0 GiB | 2.1 GiB | 1.1 GiB |
| 262,144 | 8.0 GiB | 4.25 GiB | 2.25 GiB |

Nếu cả 32 tầng đều là full attention, các con số trên nhân 4: đây là lý do kiến trúc hybrid là điều kiện cần cho context 262K trên thiết bị.

**Trạng thái GDN** không phụ thuộc độ dài context. Mỗi tầng, mỗi sequence giữ một ma trận $n_v \times d_k \times d_v$ ở F32:

$$M_{\text{GDN}} = 24 \times 32 \times 128 \times 128 \times 4 \text{ byte} = 48 \text{ MiB}$$

cộng với trạng thái conv rất nhỏ ($3 \times 8192 \times 4$ byte mỗi tầng). Mỗi sequence song song (`-np`) cần một bản riêng.

**Tổng ước lượng** cho GPU 12 GB, context 131K, KV q8_0: $2.42$ (trọng số `PQ2_0`) $+ 2.1$ (KV) $+ 0.05$ (GDN) $+$ khoảng $1$ đến $1.5$ GB buffer compute, cộng $0.62$ GB nếu nạp mmproj. Vừa khít với dư địa an toàn.

```bash
./prism/build/bin/llama-server -m models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  --mmproj models/Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf \
  -ngl 99 -fa on -c 131072 -ctk q8_0 -ctv q8_0 -np 1 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.0 --repeat-penalty 1.0 --repeat-last-n 64
```

Các giá trị lấy mẫu là khuyến nghị cho chế độ thinking trong model card; `presence_penalty` không có key GGUF tương ứng nên phải truyền qua dòng lệnh.

## Phần 5: Đo throughput

```bash
for f in Q2_0 PQ2_0 PTQ1_0; do
  ./prism/build/bin/llama-bench -m models/Ternary-Bonsai-2-9B-$f.gguf -ngl 99 -fa on -p 512 -n 128 -r 5
done
./mainline/build/bin/llama-bench -m models/Ternary-Bonsai-2-9B-Q2_0.gguf -ngl 99 -fa on -p 512 -n 128 -r 5
```

Số tham chiếu do PrismML công bố (tg128, batch 1, gói nhanh nhất mỗi thiết bị): RTX 4090 248.6 tok/s với `PTQ1_0`; MacBook Pro M5 Pro 76.5 tok/s với `PQ2_0` trên Metal; iPhone 17 Pro Max 22.3 tok/s. Ghi số đo của bạn vào bảng:

| Runtime / gói | pp512 (tok/s) | tg128 (tok/s) | Trọng số / bước (GB) | Băng thông hiệu dụng tg128 $\times$ GB |
|:---|:---:|:---:|:---:|:---:|
| fork `PTQ1_0` | | | 2.00 | |
| fork `PQ2_0` | | | 2.42 | |
| fork `Q2_0` | | | 2.56 | |
| mainline `Q2_0` | | | 2.56 | |

Cột cuối so với băng thông danh định của GPU cho biết decode còn bị chặn bởi băng thông hay đã chuyển sang bị chặn bởi giải nén và khởi chạy kernel ([Bài 11](../lesson_11_bonsai_model_family), Phần 5).

## Phần 6: Speculative decoding

```bash
# MTP: nạp file có đầu MTP thay cho file PQ2_0 thường
./prism/build/bin/llama-server -m models/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf -ngl 99 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0

# DFlash 2: drafter khối riêng, có lợi với tác vụ chép nhiều (sửa code, viết lại)
./prism/build/bin/llama-server -m models/Ternary-Bonsai-2-9B-PQ2_0.gguf \
  -md models/Ternary-Bonsai-2-9B-DFlash2-Q8_0.gguf -ngl 99 -ngld 99 --spec-type draft-dflash
```

Theo whitepaper Bonsai 2 Small, MTP kèm prompt lookup là chế độ tốt nhất hoặc gần tốt nhất ở hầu hết cấu hình (K = 2 trên CUDA, K = 3 trên M5 Pro), với tăng tốc 1.3x đến 1.4x cho prompt thông thường và 1.7x cho prompt chép nhiều trên RTX 4090. Độ sâu nháp nhỏ là hợp lý: một bước decode của mô hình 1.75 đến 2.1 bit vốn đã rất rẻ, nên chi phí verify và rollback trạng thái GDN ([Bài 12](../lesson_12_prism_llama_cpp_fork), Phần 6) nhanh chóng ăn hết lợi ích khi K lớn. Trên CUDA, đầu ra speculative có thể khác đầu ra thường ở các vị trí hai token gần đồng xác suất, do khác biệt thứ tự cộng số thực giữa batch verify và batch 1.

## Phần 7: Bảng chẩn đoán

| Triệu chứng | Nguyên nhân | Cách xử lý |
|:---|:---|:---|
| `unknown type` hoặc từ chối nạp `PQ2_0`, `PTQ1_0` | Runtime là mainline | Dùng fork, hoặc dùng file `Q2_0` |
| Nạp `Q2_0` thành công nhưng văn bản vô nghĩa | Runtime cũ, chưa có runtime Hadamard | Cập nhật mainline hoặc dùng fork |
| Lỗi "legacy Prism Q2_0 layout" | File Ternary Bonsai thế hệ 1 cũ (group 128 lưu dưới id 42) | Tải file `PQ2_0` hoặc `_g64` |
| Câu trả lời rỗng hoặc bị cắt | Giới hạn `-n` hay `max_tokens` quá nhỏ cho phần thinking | Đặt từ 8192 trở lên, hoặc tắt thinking qua `chat_template_kwargs` |
| "CPU model buffer size" vài GB dù báo offload hết | Backend GPU không có kernel cho gói đó | Đổi gói (ví dụ `PTQ1_0` trên Vulkan) hoặc build fork mới nhất |
| Tự lượng tử mô hình của mình sang `PQ2_0` cho ra rác | Bộ lượng tử tham chiếu chỉ lossless với trọng số đã ternary và đã xoay | Dùng các kiểu K-quant hoặc I-quant thông thường |
| BF16 mmproj làm server crash khi nhiều request ảnh | Lỗi đã biết của fork | Dùng mmproj Q8_0 |

## Phần 8: Bài học rút ra

1. Với mô hình có cơ sở xoay, **metadata là một phần của mô hình**: kiểm tra `prism.hadamard.*` trước khi đánh giá chất lượng.
2. Hybrid attention thay đổi bài toán bộ nhớ: KV Cache chỉ tăng ở một phần tư số tầng, trạng thái hồi quy là hằng số theo context.
3. Ba gói của cùng một mô hình chứa cùng trọng số; chọn gói là chọn sự đánh đổi giữa băng thông và chi phí giải nén trên phần cứng cụ thể, nên phải đo chứ không suy ra từ bpw.

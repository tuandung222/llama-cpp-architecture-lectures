# Bài 3: Benchmark Suite - Đo hiệu năng toàn diện

## Phần 1. Mục tiêu

Thiết kế và chạy benchmark suite toàn diện để so sánh hiệu năng giữa các cấu hình quantization, context length, số thread, và backend. Kết quả dùng để đưa ra quyết định chọn cấu hình tối ưu cho use case cụ thể.

---

## Phần 2. llama-bench: Công cụ benchmark chính thức

### Chạy benchmark cơ bản

```bash
# So sánh nhiều model và nhiều mức thread trong một lần chạy
./build/bin/llama-bench \
    -m model-q4_k_m.gguf \
    -m model-q5_k_m.gguf \
    -m model-q8_0.gguf \
    -p 512 -n 128 \
    -t 1,4,8 \
    -r 3
```

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model file (lặp lại cờ hoặc phân cách bằng dấu phẩy) |
| `-p` | Prompt processing (prefill) token count, mặc định 512 |
| `-n` | Generation (decode) token count, mặc định 128 |
| `-pg pp,tg` | Test kết hợp prefill rồi decode |
| `-d` | Độ sâu context: prefill trước `d` token vào KV cache rồi mới đo |
| `-t` | Thread count (danh sách, hoặc dải `first-last`) |
| `-r` | Số lần lặp, mặc định 5; kết quả là trung bình ± độ lệch chuẩn |

Mọi tham số (trừ `-r`, `-o`, `-v`) đều nhận nhiều giá trị, và `llama-bench` chạy **tích Descartes** của chúng: ví dụ trên tạo $3 \times 3 \times 2 = 18$ test. Thời gian tokenize và sampling không được tính.

### Output mẫu

```
| model              | size     | params | backend    | threads | test       | t/s          |
|--------------------|----------|--------|------------|---------|------------|--------------|
| llama 8B Q4_K_M    | 4.58 GiB | 8.03 B | CPU        | 4       | pp512      | 45.23 ± 0.31 |
| llama 8B Q4_K_M    | 4.58 GiB | 8.03 B | CPU        | 4       | tg128      | 12.87 ± 0.08 |
| llama 8B Q5_K_M    | 5.33 GiB | 8.03 B | CPU        | 4       | pp512      | 42.11 ± 0.28 |
| llama 8B Q5_K_M    | 5.33 GiB | 8.03 B | CPU        | 4       | tg128      | 11.23 ± 0.05 |
| llama 8B Q8_0      | 7.95 GiB | 8.03 B | CPU        | 4       | pp512      | 38.45 ± 0.19 |
| llama 8B Q8_0      | 7.95 GiB | 8.03 B | CPU        | 4       | tg128      | 7.82 ± 0.04  |
```

(Kích thước là của Llama 3 8B; số t/s chỉ minh họa định dạng. Nhãn test có dạng `pp512`, `tg128`, `pp512+tg128`, và thêm hậu tố `@ d4096` khi dùng `-d`.) Tỷ số tg giữa các quant nên xấp xỉ nghịch đảo tỷ số kích thước: $4.58 / 7.95 \approx 0.58$, nên nếu Q4_K_M đạt 12.9 t/s thì Q8_0 chỉ nên quanh 7 đến 8 t/s.

---

## Phần 3. Benchmark Prompt Processing vs Generation

Hai phase inference có đặc tính rất khác nhau:

```bash
# Prompt processing (prefill) - compute-bound
./build/bin/llama-bench -m model.gguf -p 128,256,512,1024,2048,4096 -n 0 -t 8 -r 5

# Generation (decode) - memory-bandwidth-bound, đo ở nhiều độ sâu context
./build/bin/llama-bench -m model.gguf -p 0 -n 128 -d 0,1024,4096,16384 -t 8 -r 5
```

Phân tích:
- **Prefill** (t/s) tăng theo kích thước batch khi GEMM tận dụng được compute (đến giới hạn `-ub`, mặc định 512), sau đó đi ngang rồi giảm dần với prompt rất dài vì chi phí attention tăng theo $O(n^2)$.
- **Decode** (t/s) giảm dần theo độ sâu context: mỗi token mới phải đọc toàn bộ KV cache, lượng byte đọc thêm tỷ lệ với $n_{ctx}$. Dùng `-d` để thấy rõ hiệu ứng này; `-n 64,128,256` ở độ sâu 0 gần như cho cùng một con số.

---

## Phần 4. Thread Scaling Benchmark

```bash
# Test thread scaling trên CPU 16-core
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -t 1,2,4,8,12,16 -r 3
```

Kết quả thường cho thấy:
- **Prefill** (compute-bound) scale gần tuyến tính đến số core vật lý.
- **Decode** (bandwidth-bound) bão hòa sớm: khi vài thread đã đủ làm đầy băng thông bộ nhớ, thêm thread không giúp gì, thậm chí làm chậm do tranh chấp.
- Hyperthreading (2N threads) thường không nhanh hơn N threads.

**Khuyến nghị**: dùng số thread = số physical cores, không dùng hyperthreading.

---

## Phần 5. GPU Offload Benchmark

```bash
# Test các mức GPU offload khác nhau
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -ngl 0,8,16,24,32,99 -r 3
```

| ngl (layers offloaded) | PP (t/s) | TG (t/s) | VRAM Usage |
|:---|:---|:---|:---|
| 0 (CPU only) | 45 | 12 | 0 GB |
| 8 | 62 | 18 | 1.2 GB |
| 16 | 85 | 28 | 2.4 GB |
| 24 | 110 | 38 | 3.6 GB |
| 32 (mọi repeating layer) | 180 | 65 | 4.2 GB |
| 99 (thêm cả output layer) | 185 | 68 | 4.8 GB |

(Số liệu minh họa cho một model 8B Q4_K_M, 32 layer.) Với model 32 layer, `-ngl 33` trở lên đã offload toàn bộ, kể cả tensor `output`; 99 chỉ là cách viết "tất cả". KV cache của mỗi layer nằm cùng thiết bị với layer đó theo mặc định, và được điều khiển riêng bằng `-nkvo`. Mặc định hiện nay của `-ngl` là `-1` (tự động chọn theo VRAM còn trống, cơ chế `--fit`). Lưu ý TG chỉ tăng mạnh khi gần như toàn bộ model nằm trên GPU: phần còn lại trên CPU trở thành nút cổ chai băng thông.

---

## Phần 6. Memory Benchmark

```bash
# Đọc log nạp model để thấy kích thước từng thành phần
./build/bin/llama-completion -m model.gguf \
    -c 2048 -t 8 \
    -p "Test" -n 1 -no-cnv \
    2>&1 | grep -E "file size|llama_kv_cache|compute buffer"
```

Output mẫu (Llama 3 8B Q4_K_M, ctx 2048):

```
print_info: file size   = 4.58 GiB (4.89 BPW)
llama_kv_cache: size =  256.00 MiB (  2048 cells,  32 layers,  1/1 seqs), K (f16):  128.00 MiB, V (f16):  128.00 MiB
sched_reserve:        CPU compute buffer size =   258.50 MiB
```

Kiểm tra bằng công thức bên dưới: $2 \times 32 \times 8 \times 128 \times 2048 \times 2 = 268{,}435{,}456$ byte, đúng 256 MiB. `llama-server` và `llama-perplexity` còn in bảng phân rã bộ nhớ theo thiết bị khi kết thúc (`common_memory_breakdown_print`).

### Memory estimation formula

$$\text{Total RAM} = \text{Model size} + \text{KV Cache} + \text{Compute buffer} + \text{Overhead}$$

KV Cache:

$$\text{KV} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{ctx\_len} \times \text{sizeof}(type)$$

---

## Phần 7. Batch Processing Benchmark

```bash
# Batch size impact trên prompt processing
./build/bin/llama-bench -m model.gguf \
    -p 512 -n 128 \
    -b 512,1024,2048 \
    -ub 512,1024 \
    -t 8 -r 3
```

| Flag | Ý nghĩa |
|:---|:---|
| `-b` | Logical batch size: số token tối đa gửi vào một lần `llama_decode` (mặc định 2048) |
| `-ub` | Physical batch size (ubatch): số token thực sự tính trong một lần chạy graph (mặc định 512) |

`-ub` lớn hơn thường tăng throughput prefill (đặc biệt trên GPU) nhưng tăng compute buffer; `-b` nhỏ hơn `-ub` không có ý nghĩa vì ubatch bị giới hạn bởi batch. Lưu ý `llama-bench` đo một chuỗi duy nhất; throughput của server với nhiều request song song cần `llama-batched-bench` hoặc tải thật lên `llama-server`.

---

## Phần 8. Export kết quả

```bash
# Output CSV cho phân tích
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -t 4,8 -r 5 -o csv > results.csv

# Output Markdown cho báo cáo
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -t 4,8 -r 5 -o md > results.md

# JSON (có kết quả từng lần lặp) hoặc SQL để nạp vào cơ sở dữ liệu
./build/bin/llama-bench -m model.gguf -o json > results.json
```

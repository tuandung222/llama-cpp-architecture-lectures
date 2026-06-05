# Bài 3: Benchmark Suite - Đo hiệu năng toàn diện

## 1. Mục tiêu

Thiết kế và chạy benchmark suite toàn diện để so sánh hiệu năng giữa các cấu hình quantization, context length, số thread, và backend. Kết quả dùng để đưa ra quyết định chọn cấu hình tối ưu cho use case cụ thể.

---

## 2. llama-bench: Công cụ benchmark chính thức

### Chạy benchmark cơ bản

```bash
# Benchmark tất cả model trong thư mục
./llama-bench \
    -m model-q4_k_m.gguf \
    -m model-q5_k_m.gguf \
    -m model-q8_0.gguf \
    -p 512 -n 128 \
    -t 1,4,8 \
    -r 3
```

| Flag | Ý nghĩa |
|:---|:---|
| `-m` | Model file (dùng nhiều lần để so sánh) |
| `-p` | Prompt processing (prefill) token count |
| `-n` | Generation (decode) token count |
| `-t` | Thread count (dùng nhiều lần) |
| `-r` | Số lần lặp (repetitions) để lấy trung bình |

### Output mẫu

```
| model              | size     | params | backend    | threads | test       | t/s          |
|--------------------|----------|--------|------------|---------|------------|--------------|
| llama 8B Q4_K_M   | 4.58 GiB | 8.03 B | CPU        | 4       | pp 512     | 45.23 ± 0.31 |
| llama 8B Q4_K_M   | 4.58 GiB | 8.03 B | CPU        | 4       | tg 128     | 12.87 ± 0.08 |
| llama 8B Q5_K_M   | 5.41 GiB | 8.03 B | CPU        | 4       | pp 512     | 42.11 ± 0.28 |
| llama 8B Q5_K_M   | 5.41 GiB | 8.03 B | CPU        | 4       | tg 128     | 11.23 ± 0.05 |
| llama 8B Q8_0      | 8.11 GiB | 8.03 B | CPU        | 4       | pp 512     | 38.45 ± 0.19 |
| llama 8B Q8_0      | 8.11 GiB | 8.03 B | CPU        | 4       | tg 128     | 9.82 ± 0.04  |
```

---

## 3. Benchmark Prompt Processing vs Generation

Hai phase inference có đặc tính rất khác nhau:

```bash
# Prompt processing (prefill) - compute-bound
./llama-bench -m model.gguf -p 128,256,512,1024,2048,4096 -n 0 -t 8 -r 5

# Generation (decode) - memory-bandwidth-bound
./llama-bench -m model.gguf -p 0 -n 64,128,256 -t 8 -r 5
```

Phân tích:
- **Prefill** tăng tuyến tính với prompt length cho đến khi hết AVX register.
- **Decode** gần như không đổi theo context length (trừ KV Cache quá lớn gây cache miss).

---

## 4. Thread Scaling Benchmark

```bash
# Test thread scaling trên CPU 16-core
for t in 1 2 4 8 12 16; do
    ./llama-bench -m model.gguf -p 512 -n 128 -t $t -r 3
done
```

Kết quả thường cho thấy:
- Speedup gần tuyến tính từ 1 đến N/2 threads (N = số core vật lý).
- Diminishing returns từ N/2 đến N threads.
- Hyperthreading (2N threads) thường chậm hơn N threads do contention.

**Khuyến nghị**: dùng số thread = số physical cores, không dùng hyperthreading.

---

## 5. GPU Offload Benchmark

```bash
# Test các mức GPU offload khác nhau
for ngl in 0 8 16 24 32 99; do
    ./llama-bench -m model.gguf -p 512 -n 128 -ngl $ngl -r 3
done
```

| ngl (layers offloaded) | PP (t/s) | TG (t/s) | VRAM Usage |
|:---|:---|:---|:---|
| 0 (CPU only) | 45 | 12 | 0 GB |
| 8 | 62 | 18 | 1.2 GB |
| 16 | 85 | 28 | 2.4 GB |
| 24 | 110 | 38 | 3.6 GB |
| 32 (all) | 180 | 65 | 4.8 GB |
| 99 (all + KV) | 185 | 68 | 5.1 GB |

---

## 6. Memory Benchmark

```bash
# Đo memory usage chi tiết
./llama-cli -m model.gguf \
    -c 2048 -t 8 \
    --verbose-prompt \
    -p "Test" -n 1 \
    2>&1 | grep -E "(model|kv|total|mem)"
```

Output mẫu:

```
llama_model_load_internal: model size  =  4.58 GiB
llama_kv_cache_init:  KV self size  =  0.50 GiB, K (f16):  0.25 GiB, V (f16):  0.25 GiB
llama_init_from_model: total size  =  5.12 GiB
```

### Memory estimation formula

$$\text{Total RAM} = \text{Model size} + \text{KV Cache} + \text{Compute buffer} + \text{Overhead}$$

KV Cache:

$$\text{KV} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{ctx\_len} \times \text{sizeof}(type)$$

---

## 7. Batch Processing Benchmark

```bash
# Batch size impact trên server workload
./llama-bench -m model.gguf \
    -p 512 -n 128 \
    -b 512,1024,2048 \
    -ub 512,1024 \
    -t 8 -r 3
```

| Flag | Ý nghĩa |
|:---|:---|
| `-b` | Batch size cho prompt processing |
| `-ub` | Physical batch size (UBatch) |

Batch size lớn hơn cải thiện throughput nhưng tăng memory usage. Sweet spot thường là 512-2048.

---

## 8. Export kết quả

```bash
# Output CSV cho phân tích
./llama-bench -m model.gguf -p 512 -n 128 -t 4,8 -r 5 --output csv > results.csv

# Output Markdown cho báo cáo
./llama-bench -m model.gguf -p 512 -n 128 -t 4,8 -r 5 --output md > results.md
```

---
sidebar_position: 4
sidebar_label: "Case 3: Triển khai trên Edge Device"
---

# Case 3: Triển khai LLM trên Edge Device (Raspberry Pi, Mobile, Embedded)

## 1. Bối cảnh

llama.cpp sinh ra với triết lý cốt lõi: chạy LLM ở mọi nơi, từ server farm đến chiếc điện thoại trong túi. Case study này phân tích cách triển khai mô hình trên các thiết bị edge với bộ nhớ hạn chế (1-8 GB RAM), không có GPU rời, và năng lượng thấp.

### Tại sao edge inference quan trọng?

- **Privacy**: dữ liệu không rời thiết bị, không cần gửi lên cloud.
- **Latency**: không có round-trip network, inference xảy ra tại chỗ.
- **Offline**: hoạt động khi không có kết nối internet.
- **Cost**: không tốn chi phí API hay GPU cloud.

---

## 2. Raspberry Pi 5 (8 GB RAM)

### Phần cứng

| Thông số | Giá trị |
|:---|:---|
| CPU | Broadcom BCM2712, 4x Cortex-A76 @ 2.4 GHz |
| RAM | 8 GB LPDDR4X |
| SIMD | ARM NEON (128-bit) |
| OS | Raspberry Pi OS 64-bit |

### Quy trình triển khai

```bash
# Build llama.cpp cho ARM NEON
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Quantize model cực nhỏ cho 8GB RAM
./llama-quantize model-f16.gguf model-q4_0.gguf Q4_0

# Chạy inference với context ngắn
./llama-cli -m model-q4_0.gguf \
    -c 512 \
    -t 4 \
    --mlock \
    -p "Translate to Vietnamese:" \
    -n 64
```

### Phân tích bộ nhớ

Với mô hình 3B parameters, quant Q4_0:

$$\text{Model size} = 3 \times 10^9 \times 4.5 \text{ bytes/param} \approx 1.5 \text{ GB}$$

KV Cache cho context 512 tokens:

$$\text{KV Cache} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{ctx} \times \text{sizeof}(f16)$$

Với 3B model (28 layers, 8 heads, 128 dim):

$$\text{KV Cache} \approx 2 \times 28 \times 8 \times 128 \times 512 \times 2 \approx 46 \text{ MB}$$

Tổng: ~1.6 GB, hoàn toàn vừa trong 8 GB RAM (trừ OS ~1 GB).

---

## 3. Android (Termux + llama.cpp)

### Cài đặt

```bash
# Cài Termux từ F-Droid (không dùng Play Store)
pkg update && pkg upgrade
pkg install cmake make clang git

# Clone và build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8

# Tải model Q4_0 nhỏ gọn
curl -LO https://huggingface.co/.../model-q4_0.gguf

# Chạy
./build/bin/llama-cli -m model-q4_0.gguf -t 4 -c 256 -n 50
```

### Tối ưu cho ARM big.LITTLE

ARM SoC thường có 2 loại nhân: performance cores (Cortex-X/A7x) và efficiency cores (Cortex-A5x). llama.cpp cần cấu hình thread đúng:

```bash
# Chỉ dùng P-cores (performance) để tránh jitter
./llama-cli -m model.gguf -t 4 --cpu-mask 0xF0
```

Flag `--cpu-mask` cho phép pin thread vào các core cụ thể. Trên Snapdragon 8 Gen 3:
- Core 0-3: Cortex-A520 (efficiency)
- Core 4-6: Cortex-A720 (performance)
- Core 7: Cortex-X4 (prime)

Chỉ nên dùng core 4-7 cho inference.

---

## 4. WASM (WebAssembly) trong trình duyệt

llama.cpp có thể biên dịch sang WebAssembly để chạy trực tiếp trong trình duyệt:

```bash
# Build cho WASM với Emscripten
emcmake cmake -B build-wasm \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF
emmake make -C build-wasm llama-cli
```

Sau đó serve file WASM + model qua HTTP, người dùng tải model vào bộ nhớ trình duyệt và inference hoàn toàn client-side.

### Hạn chế

| Yếu tố | Giới hạn |
|:---|:---|
| RAM trình duyệt | Thường 2-4 GB |
| SIMD | WASM SIMD 128-bit (không có AVX) |
| Thread | Web Workers (giới hạn) |
| Tốc độ | Chậm hơn native ~3-5x |

---

## 5. Checklist tối ưu Edge

1. **Chọn model nhỏ**: Qwen2.5-1.5B, Llama 3.2-1B, SmolLM-1.7B.
2. **Quant cực thấp**: Q4_0 hoặc Q4_K_S cho bộ nhớ tối thiểu.
3. **Context ngắn**: 256-512 tokens để tiết kiệm KV Cache.
4. **`--mlock`**: khóa bộ nhớ, tránh swap sang disk.
5. **Thread pinning**: chỉ dùng performance cores trên ARM big.LITTLE.
6. **Batch size 1**: edge không cần batch processing.

## 6. Kết luận

llama.cpp cho phép chạy LLM trên mọi thiết bị có CPU, từ Raspberry Pi đến điện thoại Android. Chìa khóa là chọn đúng model size, quantization level, và tối ưu thread cho kiến trúc phần cứng cụ thể. Edge inference mở ra khả năng privacy-first AI mà không cần cloud infrastructure.

---
sidebar_position: 4
sidebar_label: "Case 3: Triển khai trên Edge Device"
---

# Case 3: Triển khai LLM trên Edge Device (Raspberry Pi, Mobile, Embedded)

## Phần 1. Bối cảnh

llama.cpp sinh ra với triết lý cốt lõi: chạy LLM ở mọi nơi, từ server farm đến chiếc điện thoại trong túi. Case study này phân tích cách triển khai mô hình trên các thiết bị edge với bộ nhớ hạn chế (1-8 GB RAM), không có GPU rời, và năng lượng thấp.

### Tại sao edge inference quan trọng?

- **Privacy**: dữ liệu không rời thiết bị, không cần gửi lên cloud.
- **Latency**: không có round-trip network, inference xảy ra tại chỗ.
- **Offline**: hoạt động khi không có kết nối internet.
- **Cost**: không tốn chi phí API hay GPU cloud.

---

## Phần 2. Raspberry Pi 5 (8 GB RAM)

### Phần cứng

| Thông số | Giá trị |
|:---|:---|
| CPU | Broadcom BCM2712, 4x Cortex-A76 @ 2.4 GHz |
| RAM | 8 GB LPDDR4X |
| SIMD | ARM NEON (128-bit) |
| OS | Raspberry Pi OS 64-bit |

### Quy trình triển khai

```bash
# Build native trên chính Pi 5: GGML_NATIVE=ON (mặc định) tự bật NEON, dotprod
# và các mở rộng ARMv8.2 mà Cortex-A76 hỗ trợ
sudo apt install -y build-essential cmake git libssl-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4

# Quantize model cho 8GB RAM (Q4_0 có đường repack tối ưu cho ARM)
./build/bin/llama-quantize model-f16.gguf model-q4_0.gguf Q4_0

# Chạy một lượt với context ngắn, khóa model trong RAM
./build/bin/llama-cli -m model-q4_0.gguf \
    -c 512 \
    -t 4 \
    --load-mode mmap+mlock \
    -p "Translate to Vietnamese: Good morning" \
    -n 64 -st
```

Hai điểm khác với các hướng dẫn cũ: (1) `-DGGML_CPU_ALL_VARIANTS=ON` chỉ dùng khi phân phối một binary cho nhiều loại CPU và bắt buộc đi kèm `-DGGML_BACKEND_DL=ON`, không cần khi build tại chỗ; (2) cờ `--mlock` cũ đã được thay bằng `-lm, --load-mode` với các giá trị `auto|none|mmap|mlock|mmap+mlock|dio`.

### Phân tích bộ nhớ

Với mô hình 3B parameters, quant Q4_0:

$$\text{Model size} \approx 3 \times 10^9 \times \frac{4.5 \text{ bits}}{8 \text{ bits/byte}} \approx 1.69 \text{ GB}$$

(Q4_0 dùng 18 byte cho mỗi block 32 weight, tức 4.5 bits/param, không phải 4.5 bytes/param.)

KV Cache cho context 512 tokens:

$$\text{KV Cache} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{ctx} \times \text{sizeof}(f16)$$

Hệ số 2 ứng với K và V; với GQA, số head cần lưu là $n_{kv\_heads}$ chứ không phải số query head. Với một model 3B kiểu Llama 3.2 (28 layers, 8 KV heads, $d_{head} = 128$):

$$\text{KV Cache} = 2 \times 28 \times 8 \times 128 \times 512 \times 2 = 58{,}720{,}256 \text{ bytes} \approx 58.7 \text{ MB} \; (56 \text{ MiB})$$

Tổng: khoảng 1.75 GB cộng compute buffer (vài trăm MB), hoàn toàn vừa trong 8 GB RAM (trừ OS khoảng 1 GB).

---

## Phần 3. Android (Termux + llama.cpp)

### Cài đặt

```bash
# Termux: bản F-Droid / GitHub release là kênh ổn định nhất
pkg update && pkg upgrade
pkg install git cmake clang libandroid-spawn

# Clone và build
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8

# Tải model Q4_0 nhỏ gọn (đặt trong ~/ để đọc nhanh hơn)
curl -L -o ~/model-q4_0.gguf https://huggingface.co/<user>/<repo>/resolve/main/model-q4_0.gguf

# Chạy (hoặc tải trực tiếp: -hf <user>/<repo>:Q4_0)
./build/bin/llama-cli -m ~/model-q4_0.gguf -t 4 -c 2048 -n 50
```

### Tối ưu cho ARM big.LITTLE

ARM SoC thường có 2 loại nhân: performance cores (Cortex-X/A7x) và efficiency cores (Cortex-A5x). llama.cpp cần cấu hình thread đúng:

```bash
# Chỉ dùng các core hiệu năng (cpu2 đến cpu7) để tránh jitter
./build/bin/llama-cli -m model.gguf -t 6 -C 0xFC --cpu-strict 1
```

Flag `-C, --cpu-mask` nhận mask hex các CPU được phép, `--cpu-strict 1` buộc mỗi thread ở đúng một CPU. Snapdragon 8 Gen 3 có cấu hình 1+5+2:
- cpu0-1: Cortex-A520 (efficiency)
- cpu2-6: Cortex-A720 (performance)
- cpu7: Cortex-X4 (prime)

Nên dùng cpu2-7 (mask `0xFC`, 6 thread) cho inference. Thứ tự đánh số có thể khác giữa các thiết bị; kiểm tra bằng `cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq` trước khi chọn mask.

---

## Phần 4. WASM (WebAssembly) trong trình duyệt và WASI

llama.cpp có thể biên dịch sang WebAssembly, nhưng cần hiểu đúng phạm vi: khi build bằng Emscripten, `tools/CMakeLists.txt` và `examples/CMakeLists.txt` **bỏ qua** các chương trình như `llama-cli`, chỉ các thư viện (`libllama`, `libggml`) được build. Ứng dụng web phải tự viết lớp binding JavaScript (hoặc dùng một dự án cộng đồng xây sẵn trên các thư viện này).

```bash
# Thư viện cho trình duyệt với Emscripten (cấu hình tương tự CI build-wasm.yml)
source emsdk/emsdk_env.sh
emcmake cmake -B build-wasm \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF \
    -DLLAMA_OPENSSL=OFF
cmake --build build-wasm --target llama -j
```

Với Emscripten, build mặc định dùng 64-bit memory (`LLAMA_WASM_MEM64=ON`, cờ `-sMEMORY64=1`) và `-sALLOW_MEMORY_GROWTH=1`. Muốn tăng tốc bằng GPU trong trình duyệt, thêm `-DGGML_WEBGPU=ON` và gói emdawnwebgpu theo `docs/build.md`.

Ngoài trình duyệt, `docs/build.md` mô tả đường **WASI** bằng wasi-sdk: build thư viện với toolchain wasi-sdk, rồi tự link `examples/simple/simple.cpp` thành `llama-simple.wasm` và chạy bằng `wasmtime`.

### Hạn chế

| Yếu tố | Giới hạn |
|:---|:---|
| RAM trình duyệt | Thường 2-4 GB |
| SIMD | WASM SIMD 128-bit (không có AVX) |
| Thread | Web Workers (giới hạn) |
| Tốc độ | Chậm hơn native đáng kể (ước lượng vài lần, phụ thuộc SIMD và số thread) |

---

## Phần 5. Checklist tối ưu Edge

1. **Chọn model nhỏ**: Qwen2.5-1.5B, Llama 3.2-1B, SmolLM2-1.7B; với RAM rất hạn chế, cân nhắc model vốn đã 1-bit/ternary như BitNet b1.58 2B4T ([Bài 10](../lesson_10_bitnet_cpp)) hoặc Bonsai ([Bài 11](../lesson_11_bonsai_model_family)).
2. **Quant thấp**: Q4_0 (có đường repack tối ưu cho ARM) hoặc Q4_K_S/IQ4_XS cho bộ nhớ tối thiểu.
3. **Context ngắn**: 512 đến 2048 tokens để tiết kiệm KV Cache; có thể lượng tử KV bằng `-ctk q8_0 -ctv q8_0`.
4. **`--load-mode mmap+mlock`**: khóa model trong RAM, tránh bị swap hoặc nén bộ nhớ.
5. **Thread pinning**: chỉ dùng performance cores trên ARM big.LITTLE (`-C`, `--cpu-strict`).
6. **Một chuỗi**: edge thường chỉ phục vụ một người dùng, giữ `-np 1`.

## Phần 6. Kết luận

llama.cpp cho phép chạy LLM trên mọi thiết bị có CPU, từ Raspberry Pi đến điện thoại Android. Chìa khóa là chọn đúng model size, quantization level, và tối ưu thread cho kiến trúc phần cứng cụ thể. Edge inference mở ra khả năng privacy-first AI mà không cần cloud infrastructure.

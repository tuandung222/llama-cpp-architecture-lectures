# Bài 1: Build llama.cpp từ Source Code

## 1. Mục tiêu

Xây dựng llama.cpp từ source code với hỗ trợ đầy đủ cho CPU (AVX2/AVX512), CUDA, Metal, và Vulkan. Hiểu được hệ thống build CMake, các compile flags, và cách chọn backend.

---

## 2. Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git \
    libcurl4-openssl-dev

# CUDA (nếu có NVIDIA GPU)
# Cài CUDA Toolkit 12.x từ https://developer.nvidia.com/cuda-downloads

# macOS
xcode-select --install
brew install cmake
```

---

## 3. Clone và Build cơ bản (CPU only)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build mặc định (CPU, AVX2)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Kiểm tra build thành công

```bash
./build/bin/llama-cli --version
# Output: version: xxxx (commit xxxxxxxx)

# Kiểm tra SIMD support
./build/bin/llama-cli --help 2>&1 | head -5
```

---

## 4. Build với CUDA (NVIDIA GPU)

```bash
cmake -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

cmake --build build-cuda --config Release -j$(nproc)
```

| Architecture Flag | GPU |
|:---|:---|
| 75 | Turing (RTX 20xx, T4) |
| 80 | Ampere (A100) |
| 86 | Ampere (RTX 30xx) |
| 89 | Ada Lovelace (RTX 40xx) |

Kiểm tra CUDA support:

```bash
./build-cuda/bin/llama-cli -m model.gguf -p "Hello" -n 5 -ngl 99
# Log phải hiện: "using CUDA" và tên GPU
```

---

## 5. Build với Metal (macOS Apple Silicon)

```bash
cmake -B build-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON

cmake --build build-metal --config Release -j$(sysctl -n hw.ncpu)
```

Metal sẽ tự động sử dụng GPU Apple Silicon (M1/M2/M3/M4) qua MPS (Metal Performance Shaders).

---

## 6. Build với Vulkan (AMD/Intel GPU)

```bash
# Cài Vulkan SDK trước
cmake -B build-vulkan \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON

cmake --build build-vulkan --config Release -j$(nproc)
```

---

## 7. CMake Options quan trọng

| Option | Mặc định | Ý nghĩa |
|:---|:---|:---|
| `GGML_CUDA` | OFF | Bật CUDA backend |
| `GGML_METAL` | OFF | Bật Metal backend |
| `GGML_VULKAN` | OFF | Bật Vulkan backend |
| `GGML_NATIVE` | ON | Tối ưu cho CPU hiện tại |
| `GGML_AVX512` | OFF | Bật AVX-512 (server CPU) |
| `GGML_BLAS` | OFF | Bật external BLAS (OpenBLAS, MKL) |
| `LLAMA_CURL` | ON | Bật download model từ URL |
| `BUILD_SHARED_LIBS` | OFF | Build shared library (.so/.dylib) |

---

## 8. BLAS Acceleration (CPU)

Khi không có GPU, external BLAS giúp tăng tốc matrix multiplication đáng kể:

```bash
# OpenBLAS
sudo apt install -y libopenblas-dev

cmake -B build-blas \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build-blas --config Release -j$(nproc)
```

Intel MKL cho kết quả tốt hơn trên Intel CPU:

```bash
cmake -B build-mkl \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Intel10_64lp
```

---

## 9. Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|:---|:---|:---|
| `CUDA not found` | Thiếu CUDA toolkit | Cài CUDA hoặc thêm `-DCUDAToolkit_ROOT=/path` |
| `unsupported GNU version` | GCC quá mới cho nvcc | Dùng `-DCMAKE_CUDA_COMPILER=gcc-11` |
| `AVX2 not supported` | CPU cũ | Thêm `-DGGML_AVX=ON -DGGML_AVX2=OFF` |
| Metal shader compile error | macOS cũ | Update Xcode Command Line Tools |

---

## 10. Kiểm tra hoàn chỉnh

```bash
# Tải model test nhỏ
curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Chạy test
./build/bin/llama-cli \
    -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
    -p "Hello, how are you?" \
    -n 50 -t 4

# Verify: phải sinh ra text hợp lý, tốc độ > 10 t/s trên CPU hiện đại
```

# Bài 1: Build llama.cpp từ Source Code

## Phần 1. Mục tiêu

Xây dựng llama.cpp từ source code với hỗ trợ đầy đủ cho CPU (AVX2/AVX512), CUDA, Metal, và Vulkan. Hiểu được hệ thống build CMake, các compile flags, và cách chọn backend.

---

## Phần 2. Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git \
    libssl-dev
# libssl-dev: tùy chọn, bật HTTPS để tải model bằng -hf (LLAMA_OPENSSL=ON mặc định).
# libcurl không còn cần thiết: LLAMA_CURL đã deprecated.

# CUDA (nếu có NVIDIA GPU)
# Cài CUDA Toolkit 12.x từ https://developer.nvidia.com/cuda-downloads

# macOS
xcode-select --install
brew install cmake
```

---

## Phần 3. Clone và Build cơ bản (CPU only)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build mặc định: CPU backend, GGML_NATIVE=ON tự dò tập lệnh (AVX2, AVX-512, NEON...)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Kiểm tra build thành công

```bash
./build/bin/llama-cli --version
# Output: version: xxxx (commit xxxxxxxx)

# Liệt kê các device/backend mà binary nhìn thấy
./build/bin/llama-cli --list-devices

# Kiểm tra SIMD: llama-completion in dòng "system_info: ... AVX2 = 1 | ..." khi khởi động
./build/bin/llama-completion -m model.gguf -p "Hi" -n 1 -no-cnv 2>&1 | grep system_info
```

---

## Phần 4. Build với CUDA (NVIDIA GPU)

```bash
cmake -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

cmake --build build-cuda --config Release -j$(nproc)
```

Nếu bỏ `CMAKE_CUDA_ARCHITECTURES`, với `GGML_NATIVE=ON` CMake build cho GPU đang gắn trên máy; liệt kê tường minh khi cần một binary chạy trên nhiều thế hệ GPU.

| Architecture Flag | GPU |
|:---|:---|
| 75 | Turing (RTX 20xx, T4) |
| 80 | Ampere (A100) |
| 86 | Ampere (RTX 30xx, A10) |
| 89 | Ada Lovelace (RTX 40xx, L4) |
| 90 | Hopper (H100, H200) |
| 120 | Blackwell consumer (RTX 50xx) |

Kiểm tra CUDA support:

```bash
./build-cuda/bin/llama-cli --list-devices
# Phải liệt kê CUDA0 kèm tên GPU và VRAM
./build-cuda/bin/llama-cli -m model.gguf -p "Hello" -n 5 -ngl 99 -st
# Log nạp model có dòng "using device CUDA0 (...)"; -ngl mặc định là auto, 99 ép offload toàn bộ
```

---

## Phần 5. Build với Metal (macOS Apple Silicon)

```bash
# Trên macOS, GGML_METAL mặc định là ON (cùng GGML_BLAS với Accelerate)
cmake -B build-metal -DCMAKE_BUILD_TYPE=Release

cmake --build build-metal --config Release -j$(sysctl -n hw.ncpu)
```

Metal backend (`ggml/src/ggml-metal/`) dùng các compute kernel viết bằng Metal Shading Language của chính GGML, không dựa vào MPS (Metal Performance Shaders), và chạy trên GPU Apple Silicon (M1 trở về sau). Muốn ép chạy CPU, build với `-DGGML_METAL=OFF` hoặc dùng `--device none` lúc chạy.

---

## Phần 6. Build với Vulkan (AMD/Intel GPU)

```bash
# Cài Vulkan SDK trước
cmake -B build-vulkan \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON

cmake --build build-vulkan --config Release -j$(nproc)
```

---

## Phần 7. CMake Options quan trọng

| Option | Mặc định | Ý nghĩa |
|:---|:---|:---|
| `GGML_CUDA` | OFF | Bật CUDA backend |
| `GGML_METAL` | ON trên macOS, OFF nơi khác | Bật Metal backend |
| `GGML_VULKAN` | OFF | Bật Vulkan backend |
| `GGML_NATIVE` | ON (OFF khi cross-compile) | Tối ưu cho CPU hiện tại (`-march=native`) |
| `GGML_AVX2`, `GGML_AVX512`... | Chỉ có hiệu lực khi `GGML_NATIVE=OFF` | Chọn tập lệnh thủ công |
| `GGML_BLAS` | ON trên macOS (Accelerate), OFF nơi khác | Bật external BLAS (OpenBLAS, MKL, BLIS...) |
| `GGML_BACKEND_DL` | OFF | Build backend thành thư viện nạp động (cần `BUILD_SHARED_LIBS`) |
| `GGML_CPU_ALL_VARIANTS` | OFF | Build mọi biến thể CPU, chọn lúc chạy (cần `GGML_BACKEND_DL=ON`) |
| `LLAMA_OPENSSL` | ON | HTTPS cho tính năng tải model (`-hf`) |
| `LLAMA_CURL` | (deprecated) | Chỉ còn sinh cảnh báo; không cần nữa |
| `BUILD_SHARED_LIBS` | ON (OFF với MinGW, Emscripten) | Build shared library (.so/.dylib) |

---

## Phần 8. BLAS Acceleration (CPU)

Khi không có GPU, external BLAS có thể tăng tốc **prompt processing** (GEMM với batch lớn); pha decode (GEMV, bị giới hạn bởi bandwidth) gần như không hưởng lợi:

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
source /opt/intel/oneapi/setvars.sh
cmake -B build-mkl \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Intel10_64lp \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_NATIVE=ON
cmake --build build-mkl --config Release -j$(nproc)
```

---

## Phần 9. Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|:---|:---|:---|
| `CUDA not found` | Thiếu CUDA toolkit | Cài CUDA hoặc thêm `-DCUDAToolkit_ROOT=/path` |
| `unsupported GNU version` | GCC quá mới cho nvcc | Chỉ định host compiler cũ hơn: `-DCMAKE_CUDA_HOST_COMPILER=g++-12` (hoặc biến môi trường `CUDAHOSTCXX`) |
| `Illegal instruction` khi chạy trên máy khác | Binary build với `GGML_NATIVE=ON` trên CPU mới hơn | Build lại trên máy đích, hoặc `-DGGML_NATIVE=OFF -DGGML_AVX2=OFF ...`, hoặc `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` |
| Metal shader compile error | macOS cũ | Update Xcode Command Line Tools |

---

## Phần 10. Kiểm tra hoàn chỉnh

```bash
# Cách 1: tải trực tiếp từ Hugging Face (cần build có OpenSSL)
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello, how are you?" -n 50 -t 4 -st

# Cách 2: dùng file đã tải về
./build/bin/llama-cli -m model.gguf -p "Hello, how are you?" -n 50 -t 4 -st

# Đo tốc độ chuẩn hóa
./build/bin/llama-bench -m model.gguf

# Verify: phải sinh ra text hợp lý; với model 1B Q4 trên CPU desktop hiện đại,
# tg128 của llama-bench thường đạt vài chục token/s
```

---
sidebar_position: 11
sidebar_label: "Bài 9: Thực hành - Tự viết GGUF Reader và Quantization"
---

# Bài 9: Thực hành - Tự viết GGUF Reader và Quantization

Không có cách nào hiểu sâu kiến trúc llama.cpp tốt hơn việc **tự tay xây dựng** các thành phần cốt lõi. Trong bài thực hành này, chúng ta sẽ viết một chương trình Python hoàn chỉnh thực hiện ba nhiệm vụ:

1. **Đọc file GGUF** từ đầu (parse header, metadata, tensor info).
2. **Implement Q8_0 quantization và dequantization**.
3. **So sánh chất lượng** trọng số giữa bản gốc FP16 và bản Q8_0.

---

## 1. GGUF Reader từ đầu (Pure Python)

```python
import struct
import numpy as np
from typing import Dict, List, Tuple, Optional

# GGUF type constants (from gguf.h)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types (from ggml.h)
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

class GGUFReader:
    """Đọc file GGUF từ đầu bằng struct module, không dùng thư viện gguf-py"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, any] = {}
        self.tensors: List[Dict] = []

        with open(filepath, 'rb') as f:
            self._read_header(f)
            self._read_metadata(f)
            self._read_tensor_info(f)

    def _read_header(self, f):
        """Đọc header: magic, version, n_tensors, n_kv"""
        magic = f.read(4)
        assert magic == b'GGUF', f"Invalid magic: {magic}"

        self.version = struct.unpack('<I', f.read(4))[0]
        self.n_tensors = struct.unpack('<q', f.read(8))[0]
        self.n_kv = struct.unpack('<q', f.read(8))[0]
        print(f"GGUF v{self.version}: {self.n_tensors} tensors, {self.n_kv} KV pairs")

    def _read_string(self, f) -> str:
        """Đọc string: uint64_t length + raw bytes"""
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8')

    def _read_value(self, f, vtype: int):
        """Đọc giá trị theo type"""
        if vtype == GGUF_TYPE_UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif vtype == GGUF_TYPE_INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif vtype == GGUF_TYPE_UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif vtype == GGUF_TYPE_INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif vtype == GGUF_TYPE_UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif vtype == GGUF_TYPE_INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif vtype == GGUF_TYPE_BOOL:
            return struct.unpack('<?', f.read(1))[0]
        elif vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif vtype == GGUF_TYPE_UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif vtype == GGUF_TYPE_INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif vtype == GGUF_TYPE_FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif vtype == GGUF_TYPE_ARRAY:
            arr_type = struct.unpack('<i', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        return None

    def _read_metadata(self, f):
        """Đọc tất cả key-value pairs"""
        for _ in range(self.n_kv):
            key = self._read_string(f)
            vtype = struct.unpack('<i', f.read(4))[0]
            value = self._read_value(f, vtype)
            self.metadata[key] = value

    def _read_tensor_info(self, f):
        """Đọc thông tin tất cả tensors"""
        for _ in range(self.n_tensors):
            name = self._read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
            tensor_type = struct.unpack('<i', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]

            self.tensors.append({
                'name': name,
                'dims': dims,
                'type': tensor_type,
                'offset': offset,
            })

    def print_summary(self):
        """In tóm tắt nội dung file GGUF"""
        print(f"\n{'='*60}")
        print(f"Metadata ({self.n_kv} entries):")
        for key, val in list(self.metadata.items())[:10]:
            val_str = str(val)[:80]
            print(f"  {key}: {val_str}")
        if self.n_kv > 10:
            print(f"  ... and {self.n_kv - 10} more entries")

        print(f"\nTensors ({self.n_tensors}):")
        type_names = {0: 'F32', 1: 'F16', 2: 'Q4_0', 8: 'Q8_0', 12: 'Q4_K', 13: 'Q5_K'}
        for t in self.tensors[:5]:
            tname = type_names.get(t['type'], f"type_{t['type']}")
            print(f"  {t['name']}: shape={t['dims']}, type={tname}")
        if self.n_tensors > 5:
            print(f"  ... and {self.n_tensors - 5} more tensors")


# Sử dụng
reader = GGUFReader("llama-3-8b-q4_k_m.gguf")
reader.print_summary()
```

---

## 2. Q8_0 Quantization và Dequantization

Q8_0 là quantization đơn giản nhất để implement: mỗi block 32 giá trị FP32 được nén thành 1 scale (F16) + 32 int8 values.

```python
import numpy as np

QK8_0 = 32  # Block size cho Q8_0

def quantize_q8_0(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize FP32/FP16 weights sang Q8_0 format.

    Args:
        weights: 1D array FP32 weights, độ dài phải chia hết cho 32

    Returns:
        scales: Array F16 scales, shape (n_blocks,)
        quantized: Array int8 quantized values, shape (n_blocks, 32)
    """
    assert weights.size % QK8_0 == 0, f"Weights size must be divisible by {QK8_0}"
    n_blocks = weights.size // QK8_0

    # Reshape thành blocks
    blocks = weights.reshape(n_blocks, QK8_0).astype(np.float32)

    # Tính scale cho mỗi block: max(|x|) / 127
    max_abs = np.max(np.abs(blocks), axis=1)
    scales = (max_abs / 127.0).astype(np.float16)

    # Tránh chia cho 0
    safe_scales = np.where(scales == 0, 1.0, scales.astype(np.float32))

    # Quantize: round(x / scale), clip về [-128, 127]
    quantized = np.round(blocks / safe_scales[:, np.newaxis])
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    return scales, quantized


def dequantize_q8_0(scales: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    """Dequantize Q8_0 về FP32.

    Args:
        scales: F16 scales, shape (n_blocks,)
        quantized: int8 values, shape (n_blocks, 32)

    Returns:
        Dequantized FP32 weights, shape (n_blocks * 32,)
    """
    scales_f32 = scales.astype(np.float32)
    dequantized = quantized.astype(np.float32) * scales_f32[:, np.newaxis]
    return dequantized.flatten()


# ============================================================
# Test: Quantize và Dequantize một weight matrix
# ============================================================
def test_q8_0_quantization():
    np.random.seed(42)

    # Tạo weight matrix giả lập (giống attention weight)
    original = np.random.randn(256).astype(np.float32) * 0.02

    # Quantize
    scales, quantized = quantize_q8_0(original)
    print(f"Original shape: {original.shape}")
    print(f"Scales shape: {scales.shape} ({scales.nbytes} bytes)")
    print(f"Quantized shape: {quantized.shape} ({quantized.nbytes} bytes)")

    # Dequantize
    reconstructed = dequantize_q8_0(scales, quantized)

    # So sánh
    mse = np.mean((original - reconstructed) ** 2)
    max_error = np.max(np.abs(original - reconstructed))
    relative_error = np.mean(np.abs(original - reconstructed)) / np.mean(np.abs(original))

    print(f"\n{'='*40}")
    print(f"Quality Metrics:")
    print(f"  MSE:           {mse:.8f}")
    print(f"  Max Error:     {max_error:.6f}")
    print(f"  Relative Error: {relative_error:.4%}")
    print(f"  Compression:   {original.nbytes}B -> {scales.nbytes + quantized.nbytes}B "
          f"({(scales.nbytes + quantized.nbytes) / original.nbytes:.2%})")

    # In vài giá trị so sánh
    print(f"\nSample comparison (first 8 values):")
    print(f"  Original:    {original[:8]}")
    print(f"  Quantized:   {quantized[0]}")
    print(f"  Dequantized: {reconstructed[:8]}")

test_q8_0_quantization()
```

---

## 3. So sánh chất lượng Multi-Quant

```python
def compare_quantization_quality(weights: np.ndarray):
    """So sánh MSE giữa các mức quantization khác nhau"""
    results = {}

    # Baseline: FP16
    fp16 = weights.astype(np.float16).astype(np.float32)
    results['FP16'] = np.mean((weights - fp16) ** 2)

    # Q8_0
    scales, quantized = quantize_q8_0(weights)
    recon = dequantize_q8_0(scales, quantized)
    results['Q8_0'] = np.mean((weights - recon) ** 2)

    # In kết quả
    print(f"\n{'Quant Type':<12} {'MSE':<15} {'Size (bytes)':<15}")
    print(f"{'='*42}")
    print(f"{'FP32':<12} {'0.0':<15} {weights.nbytes:<15}")
    for qtype, mse in results.items():
        print(f"{qtype:<12} {mse:<15.8f} {'-':<15}")

# Test với weight distribution khác nhau
weights_normal = np.random.randn(1024).astype(np.float32) * 0.02
print("Normal distribution weights:")
compare_quantization_quality(weights_normal)
```

---

## 4. Chạy Inference với Python Bindings

```python
# Cài đặt: pip install llama-cpp-python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="llama-3-8b-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,   # Offload tất cả lên GPU (nếu có)
    n_threads=8,        # CPU threads
    verbose=True,       # Xem load time và memory usage
)

# Generate text
output = llm(
    "Giải thích quantization trong LLM inference:",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    echo=False,
)

print(output['choices'][0]['text'])
print(f"\nTokens generated: {output['usage']['completion_tokens']}")
print(f"Eval time: {output.get('timings', {}).get('eval_time', 'N/A')}")
```

---

## 💡 Đúc kết Bài 9

Qua bài thực hành này, chúng ta đã:

1. **Tự viết GGUF Reader**: Parse header, metadata key-value pairs, và tensor info từ file GGUF binary.
2. **Implement Q8_0**: Quantize và dequantize, hiểu rõ cấu trúc block (scale + int8 values).
3. **Đo chất lượng**: MSE, max error, relative error cho thấy Q8_0 giữ >99% chất lượng so với FP16.
4. **Chạy inference**: Sử dụng llama-cpp-python để load model và generate text.

Chúc mừng bạn đã hoàn thành trọn vẹn chuỗi 10 bài học phân tích kiến trúc **llama.cpp**! Hy vọng kiến thức này sẽ giúp bạn làm chủ công nghệ inference LLM trên mọi nền tảng phần cứng.

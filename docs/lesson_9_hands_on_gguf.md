---
sidebar_position: 11
sidebar_label: "Bài 9: Thực hành - Tự viết GGUF Reader và Quantization"
---

# Bài 9: Thực hành - Tự viết GGUF Reader và Quantization

Không có cách nào hiểu sâu kiến trúc llama.cpp tốt hơn việc **tự tay xây dựng** các thành phần cốt lõi, rồi **đối chiếu từng bit** với cài đặt chính thức. Trong bài thực hành này, chúng ta viết một chương trình Python chỉ dùng `struct` và NumPy để:

1. **Đọc file GGUF** từ đầu: header, metadata, tensor info, tính đúng vị trí vùng dữ liệu theo alignment, và nạp dữ liệu tensor.
2. **Cài đặt Q8_0 và Q4_0** sao cho kết quả trùng từng byte với hàm tham chiếu `quantize_row_q8_0_ref` / `quantize_row_q4_0_ref` trong `ggml/src/ggml-quants.c`.
3. **Đo chất lượng** bằng MSE và SQNR, và giải thích con số đo được bằng lý thuyết.

Mọi đoạn mã dưới đây đã được chạy với NumPy 2.x và thư viện `gguf-py` của repo llama.cpp (cài bằng `pip install gguf`, hoặc `pip install -e gguf-py` từ thư mục gốc của repo để có bản mới nhất).

---

## Phần 1: Tạo một file GGUF nhỏ để thực hành

Thay vì tải một model vài GB, ta tạo một file GGUF tổng hợp bằng `gguf.GGUFWriter`, chứa ba tensor ở ba kiểu F32, F16 và Q8_0. Lợi ích kép: file nhỏ, và ta biết chính xác nội dung để kiểm tra reader.

```python
# make_tiny_gguf.py
import numpy as np
import gguf

rng = np.random.default_rng(0)
w = gguf.GGUFWriter("tiny.gguf", "llama")      # arch -> general.architecture
w.add_block_count(2)                            # khóa "llama.block_count"
w.add_context_length(4096)
w.add_embedding_length(64)
w.add_head_count(4)
w.add_string("general.name", "tiny-synthetic")
w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world"])
w.add_array("test.scores", [0.5, -1.25, 3.0])

a = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)   # numpy shape (8, 64)
b = (rng.standard_normal((8, 64)) * 0.02).astype(np.float16)
c = (rng.standard_normal((8, 64)) * 0.02).astype(np.float32)
cq = gguf.quants.quantize(c, gguf.GGMLQuantizationType.Q8_0)    # uint8, shape (8, 68)
w.add_tensor("blk.0.attn_q.weight", a)
w.add_tensor("blk.0.attn_k.weight", b)
w.add_tensor("blk.0.ffn_up.weight", cq, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
np.save("tiny_ffn_up_f32.npy", c)               # giữ bản gốc để đo sai số

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
```

Lưu ý dòng `cq`: tensor Q8_0 được truyền vào dưới dạng mảng byte shape `(8, 68)` (mỗi hàng 64 phần tử là 2 block, mỗi block 34 byte), và `GGUFWriter.add_tensor_info` dùng `quant_shape_from_byte_shape` để suy ra shape logic `(8, 64)` từ shape byte. Có thể xem nhanh file bằng công cụ `gguf-dump tiny.gguf` đi kèm gói `gguf`.

---

## Phần 2: GGUF Reader từ đầu (pure Python)

Cấu trúc file được mô tả ở đầu `ggml/include/gguf.h`: magic `"GGUF"`, version (`uint32`), số tensor và số cặp KV (`int64`), các cặp KV, các tensor info, rồi vùng dữ liệu được căn lề. Mọi enum được lưu dưới dạng `int32`, bool là `int8`, chuỗi là độ dài `uint64` theo sau là byte UTF-8 không có ký tự kết thúc.

```python
import struct
import numpy as np
from typing import Any, Dict, List

# enum gguf_type (ggml/include/gguf.h)
GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 0, 1, 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL = 4, 5, 6, 7
GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT64, GGUF_TYPE_INT64 = 8, 9, 10, 11
GGUF_TYPE_FLOAT64 = 12

# Kiểu scalar có kích thước cố định: (định dạng struct, số byte)
SCALAR_FMT = {
    GGUF_TYPE_UINT8: ('<B', 1),  GGUF_TYPE_INT8: ('<b', 1),
    GGUF_TYPE_UINT16: ('<H', 2), GGUF_TYPE_INT16: ('<h', 2),
    GGUF_TYPE_UINT32: ('<I', 4), GGUF_TYPE_INT32: ('<i', 4),
    GGUF_TYPE_FLOAT32: ('<f', 4), GGUF_TYPE_BOOL: ('<?', 1),
    GGUF_TYPE_UINT64: ('<Q', 8), GGUF_TYPE_INT64: ('<q', 8),
    GGUF_TYPE_FLOAT64: ('<d', 8),
}

# Một phần enum ggml_type (ggml/include/ggml.h): id -> (tên, block size, byte/block)
GGML_TYPES = {
    0: ('F32', 1, 4),    1: ('F16', 1, 2),    2: ('Q4_0', 32, 18),
    8: ('Q8_0', 32, 34), 12: ('Q4_K', 256, 144), 14: ('Q6_K', 256, 210),
    30: ('BF16', 1, 2),  39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36),
}

GGUF_DEFAULT_ALIGNMENT = 32


class MiniGGUFReader:
    """Đọc file GGUF bằng struct thuần, không dùng gguf-py."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Dict[str, Any]] = []
        with open(filepath, 'rb') as f:
            self._read_header(f)
            for _ in range(self.n_kv):
                key = self._read_string(f)
                vtype = struct.unpack('<i', f.read(4))[0]   # enum lưu dạng int32
                self.metadata[key] = self._read_value(f, vtype)
            for _ in range(self.n_tensors):
                self.tensors.append(self._read_tensor_info(f))
            # Vùng dữ liệu bắt đầu tại vị trí hiện tại, làm tròn lên bội số alignment
            self.alignment = int(self.metadata.get('general.alignment', GGUF_DEFAULT_ALIGNMENT))
            pos = f.tell()
            self.data_offset = (pos + self.alignment - 1) // self.alignment * self.alignment

    def _read_header(self, f):
        magic = f.read(4)
        assert magic == b'GGUF', f"Invalid magic: {magic!r}"
        self.version = struct.unpack('<I', f.read(4))[0]
        # gguf.h ghi n_tensors và n_kv là int64_t
        self.n_tensors, self.n_kv = struct.unpack('<qq', f.read(16))

    @staticmethod
    def _read_string(f) -> str:
        (length,) = struct.unpack('<Q', f.read(8))   # uint64 độ dài, không có '\0'
        return f.read(length).decode('utf-8')

    def _read_value(self, f, vtype: int):
        if vtype in SCALAR_FMT:
            fmt, size = SCALAR_FMT[vtype]
            return struct.unpack(fmt, f.read(size))[0]
        if vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        if vtype == GGUF_TYPE_ARRAY:
            arr_type = struct.unpack('<i', f.read(4))[0]
            (arr_len,) = struct.unpack('<Q', f.read(8))
            if arr_type in SCALAR_FMT:            # đọc một lần cho mảng số (nhanh hơn nhiều)
                fmt, size = SCALAR_FMT[arr_type]
                return list(struct.unpack('<' + fmt[1] * arr_len, f.read(size * arr_len)))
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        raise ValueError(f"unknown gguf_type {vtype}")

    def _read_tensor_info(self, f) -> Dict[str, Any]:
        name = self._read_string(f)
        (n_dims,) = struct.unpack('<I', f.read(4))
        ne = list(struct.unpack(f'<{n_dims}q', f.read(8 * n_dims)))  # ne[0] là chiều liên tục nhất
        ttype = struct.unpack('<i', f.read(4))[0]
        (offset,) = struct.unpack('<Q', f.read(8))   # tương đối so với data_offset
        return {'name': name, 'ne': ne, 'type': ttype, 'offset': offset}

    def tensor_nbytes(self, t: Dict[str, Any]) -> int:
        _, blck, type_size = GGML_TYPES[t['type']]
        n_el = int(np.prod(t['ne']))
        assert t['ne'][0] % blck == 0
        return n_el // blck * type_size

    def load_tensor(self, name: str) -> np.ndarray:
        """Trả về tensor FP32 có shape numpy = ne đảo ngược (row-major)."""
        t = next(t for t in self.tensors if t['name'] == name)
        with open(self.filepath, 'rb') as f:
            f.seek(self.data_offset + t['offset'])
            raw = np.frombuffer(f.read(self.tensor_nbytes(t)), dtype=np.uint8)
        tname = GGML_TYPES[t['type']][0]
        if tname == 'F32':
            flat = raw.view(np.float32)
        elif tname == 'F16':
            flat = raw.view(np.float16).astype(np.float32)
        elif tname == 'Q8_0':
            flat = dequantize_q8_0(raw)
        elif tname == 'Q4_0':
            flat = dequantize_q4_0(raw)
        else:
            raise NotImplementedError(tname)
        return flat.reshape(list(reversed(t['ne'])))

    def print_summary(self, max_items: int = 10):
        print(f"GGUF v{self.version}: {self.n_tensors} tensors, {self.n_kv} KV, "
              f"alignment={self.alignment}, data_offset={self.data_offset}")
        for key, val in list(self.metadata.items())[:max_items]:
            print(f"  {key}: {str(val)[:80]}")
        for t in self.tensors[:max_items]:
            tname = GGML_TYPES.get(t['type'], (f"type_{t['type']}",))[0]
            print(f"  {t['name']}: ne={t['ne']}, type={tname}, offset={t['offset']}")
```

Ba chi tiết thường bị làm sai khi tự viết reader:

- **Vị trí vùng dữ liệu.** `offset` trong tensor info là offset *tương đối* trong vùng dữ liệu, không phải offset tuyệt đối trong file. Vùng dữ liệu bắt đầu ngay sau tensor info cuối cùng, được làm tròn lên bội số của `general.alignment` (mặc định `GGUF_DEFAULT_ALIGNMENT = 32`). Công thức làm tròn lên $\lceil p / A \rceil \cdot A$ được viết bằng số nguyên là `(p + A - 1) // A * A`.
- **Thứ tự chiều.** `ne` là thứ tự GGML, `ne[0]` liên tục nhất; muốn có mảng NumPy row-major phải đảo ngược (Bài 5, Phần 4). Tensor `(8, 64)` của NumPy được ghi thành `ne = [64, 8]`.
- **Kích thước dữ liệu của tensor lượng tử hóa** là $\frac{n_{elements}}{\text{block size}} \cdot \text{type size}$, với block size và type size tra trong bảng (tương ứng `GGML_QUANT_SIZES` của gguf-py và `ggml_blck_size`, `ggml_type_size` của GGML). Điều kiện `ne[0] % block_size == 0` là bắt buộc: một block không bao giờ vắt qua hai hàng.

---

## Phần 3: Q8_0 và Q4_0 bit-exact

### 3.1. Q8_0

`block_q8_0` (`ggml/src/ggml-common.h`) gồm một scale `d` kiểu FP16 và 32 số `int8`, tổng 34 byte cho 32 phần tử, tức **8.5 bit mỗi trọng số**. Hàm tham chiếu:

$$d = \frac{\max_j |x_j|}{127}, \qquad q_j = \text{roundf}\left(\frac{x_j}{d}\right), \qquad \hat{x}_j = \text{fp16}(d) \cdot q_j$$

Hai chi tiết quyết định tính bit-exact: (1) `x/d` được tính với `d` ở **FP32**, chỉ khi lưu mới làm tròn `d` xuống FP16; (2) `roundf` của C làm tròn nửa **ra xa 0**, trong khi `np.round` làm tròn nửa về số chẵn (banker's rounding). Vì $|x_j / d| \le 127$, không cần clip.

```python
QK8_0 = 32
QK4_0 = 32


def roundf(x: np.ndarray) -> np.ndarray:
    """roundf của C: làm tròn nửa ra xa 0 (np.round làm tròn nửa về số chẵn)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """Bit-exact với quantize_row_q8_0_ref (ggml/src/ggml-quants.c).
    Trả về mảng uint8 shape (n_blocks, 34): 2 byte d (FP16) + 32 byte qs (int8)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, QK8_0)
    amax = np.abs(x).max(axis=1, keepdims=True)
    d = amax / np.float32(127)                          # d tính bằng FP32
    with np.errstate(divide='ignore'):
        inv_d = np.where(d == 0, np.float32(0), np.float32(1) / d)
    qs = roundf(x * inv_d).astype(np.int8)              # |x*inv_d| <= 127, không cần clip
    d16 = d.astype(np.float16)                          # chỉ lưu d dưới dạng FP16
    return np.concatenate([d16.view(np.uint8), qs.view(np.uint8)], axis=1)


def dequantize_q8_0(raw: np.ndarray) -> np.ndarray:
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, 2 + QK8_0)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)   # (n_blocks, 1)
    qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)
```

### 3.2. Q4_0

`block_q4_0` gồm `d` FP16 và 16 byte chứa 32 nibble, tổng 18 byte, **4.5 bit mỗi trọng số**. Điểm tinh tế: thay vì dùng $\max|x|/7$, hàm tham chiếu lấy giá trị $m$ có trị tuyệt đối lớn nhất **giữ nguyên dấu** và đặt $d = m / (-8)$. Khi đó phần tử cực trị được ánh xạ chính xác vào mức $-8$, mức duy nhất không có đối xứng dương trong dải $[-8, 7]$, nên cả 16 mức đều được dùng:

$$d = \frac{m}{-8}, \qquad q_j = \min\left(15, \left\lfloor \frac{x_j}{d} + 8.5 \right\rfloor\right), \qquad \hat{x}_j = d \cdot (q_j - 8)$$

Phép $\lfloor \cdot + 8.5 \rfloor$ (cast `int8` của C, tức cắt về 0, áp lên số dương) là làm tròn về gần nhất sau khi dịch dải. Layout nibble cũng đáng chú ý: byte $j$ chứa phần tử $j$ ở 4 bit thấp và phần tử $j + 16$ ở 4 bit cao, để kernel SIMD tách hai nửa block bằng một phép AND và một phép dịch.

```python
def quantize_q4_0(x: np.ndarray) -> np.ndarray:
    """Bit-exact với quantize_row_q4_0_ref: d = max/-8 (max là giá trị có |.| lớn nhất,
    giữ dấu), q = min(15, trunc(x/d + 8.5)); nibble thấp chứa phần tử j, nibble cao chứa j+16."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, QK4_0)
    imax = np.abs(x).argmax(axis=1)[:, None]
    vmax = np.take_along_axis(x, imax, axis=1)
    d = vmax / np.float32(-8)
    with np.errstate(divide='ignore'):
        inv_d = np.where(d == 0, np.float32(0), np.float32(1) / d)
    q = np.minimum(15, np.trunc(x * inv_d + np.float32(8.5))).astype(np.uint8)
    qs = q[:, :16] | (q[:, 16:] << 4)
    return np.concatenate([d.astype(np.float16).view(np.uint8), qs], axis=1)


def dequantize_q4_0(raw: np.ndarray) -> np.ndarray:
    blocks = np.asarray(raw, dtype=np.uint8).reshape(-1, 2 + QK4_0 // 2)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    return (np.concatenate([lo, hi], axis=1).astype(np.float32) * d).reshape(-1)
```

---

## Phần 4: Kiểm chứng với gguf-py

Một cài đặt "trông đúng" chưa đủ. Đoạn sau so sánh reader của ta với `gguf.GGUFReader`, và bộ lượng tử hóa của ta với `gguf.quants.quantize` (lớp `Q8_0` trong `gguf-py/gguf/quants.py` tự mô tả là "bit-exact same results as reference implementation in ggml-quants.c"):

```python
import gguf

r = MiniGGUFReader("tiny.gguf")
r.print_summary()

ref = gguf.GGUFReader("tiny.gguf")
assert ref.data_offset == r.data_offset
for t in ref.tensors:
    mine = r.load_tensor(t.name)
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        theirs = np.asarray(t.data, dtype=np.float32)
    else:
        theirs = gguf.quants.dequantize(t.data, t.tensor_type)
    assert np.array_equal(mine, theirs.reshape(mine.shape)), t.name
    print("OK", t.name, mine.shape)

rng = np.random.default_rng(42)
w = (rng.standard_normal(4096) * 0.02).astype(np.float32)
assert np.array_equal(quantize_q8_0(w),
                      gguf.quants.quantize(w.reshape(1, -1), gguf.GGMLQuantizationType.Q8_0).reshape(-1, 34))
assert np.array_equal(quantize_q4_0(w),
                      gguf.quants.quantize(w.reshape(1, -1), gguf.GGMLQuantizationType.Q4_0).reshape(-1, 18))
print("Q8_0 và Q4_0 trùng từng byte với gguf-py")
```

Kết quả chạy thực tế:

```
GGUF v3: 3 tensors, 8 KV, alignment=32, data_offset=608
  general.architecture: llama
  llama.block_count: 2
  llama.context_length: 4096
  llama.embedding_length: 64
  llama.attention.head_count: 4
  general.name: tiny-synthetic
  tokenizer.ggml.tokens: ['<s>', '</s>', 'hello', 'world']
  test.scores: [0.5, -1.25, 3.0]
  blk.0.attn_q.weight: ne=[64, 8], type=F32, offset=0
  blk.0.attn_k.weight: ne=[64, 8], type=F16, offset=2048
  blk.0.ffn_up.weight: ne=[64, 8], type=Q8_0, offset=3072
OK blk.0.attn_q.weight (8, 64)
OK blk.0.attn_k.weight (8, 64)
OK blk.0.ffn_up.weight (8, 64)
Q8_0 và Q4_0 trùng từng byte với gguf-py
```

Hãy tự kiểm tra các offset: tensor F32 chiếm $8 \cdot 64 \cdot 4 = 2048$ byte, nên tensor F16 bắt đầu ở 2048; tensor F16 chiếm 1024 byte, nên tensor Q8_0 bắt đầu ở 3072. Cả ba kích thước đều là bội của 32 nên không cần padding giữa các tensor. Thử thay `np.round` cho `roundf` trong `quantize_q8_0`: với dữ liệu ngẫu nhiên đủ lớn, assert sẽ thất bại ở đúng những phần tử có $x_j/d$ rơi vào nửa số nguyên.

---

## Phần 5: Đo và giải thích chất lượng

Ta dùng tỉ số tín hiệu trên nhiễu lượng tử hóa (SQNR):

$$\text{SQNR} = 10 \log_{10} \frac{\sum_j x_j^2}{\sum_j (x_j - \hat{x}_j)^2} \ \text{dB}$$

```python
def sqnr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    noise = np.sum((x - x_hat) ** 2)
    return float(10 * np.log10(np.sum(x ** 2) / noise)) if noise > 0 else float('inf')


def compare_quantization_quality(w: np.ndarray):
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    rows = [
        ('F16',  w.astype(np.float16).astype(np.float32), 16.0),
        ('Q8_0', dequantize_q8_0(quantize_q8_0(w)), 34 * 8 / 32),
        ('Q4_0', dequantize_q4_0(quantize_q4_0(w)), 18 * 8 / 32),
    ]
    print(f"{'type':<6} {'bpw':>6} {'MSE':>12} {'max|err|':>10} {'SQNR(dB)':>9}")
    for name, w_hat, bpw in rows:
        err = w - w_hat
        print(f"{name:<6} {bpw:>6.2f} {np.mean(err**2):>12.3e} "
              f"{np.max(np.abs(err)):>10.2e} {sqnr_db(w, w_hat):>9.2f}")


rng = np.random.default_rng(42)
w = (rng.standard_normal(4096) * 0.02).astype(np.float32)
print("normal"); compare_quantization_quality(w)
w_out = (rng.standard_normal(4096) * 0.02).astype(np.float32)
w_out[::97] *= 20                                   # chèn outlier thưa
print("outlier"); compare_quantization_quality(w_out)
```

Kết quả chạy thực tế:

```
normal
type      bpw          MSE   max|err|  SQNR(dB)
F16     16.00    1.712e-11   2.98e-05     73.66
Q8_0     8.50    1.155e-08   2.75e-04     45.37
Q4_0     4.50    2.914e-06   6.03e-03     21.36
outlier
type      bpw          MSE   max|err|  SQNR(dB)
F16     16.00    7.244e-11   1.96e-04     74.33
Q8_0     8.50    2.313e-07   3.25e-03     39.29
Q4_0     4.50    4.432e-05   4.08e-02     16.46
```

**Giải thích bằng lý thuyết.** Với bộ lượng tử hóa đều bước $\Delta$, sai số xấp xỉ phân phối đều trên $[-\Delta/2, \Delta/2]$ nên có phương sai $\Delta^2/12$. Với trọng số Gauss độ lệch chuẩn $\sigma$, giá trị $\max|x|$ của một block 32 phần tử có kỳ vọng khoảng $2.35\sigma$ (ước lượng Monte Carlo trên 200000 block). Với Q8_0, $\Delta \approx 2.35\sigma / 127$:

$$\text{SQNR}_{Q8\_0} \approx 10 \log_{10} \frac{\sigma^2}{\Delta^2/12} = 10 \log_{10}\left(12 \cdot \left(\frac{127}{2.35}\right)^2\right) \approx 45.4 \ \text{dB}$$

Với Q4_0, $\Delta \approx 2.35\sigma / 8$ cho khoảng $21.4$ dB. Hai giá trị lý thuyết này khớp với 45.37 dB và 21.36 dB đo được (mô hình $\Delta^2/12$ bỏ qua việc phần tử cực trị được biểu diễn chính xác, và FP16 hóa scale chỉ thêm sai số rất nhỏ). Hiệu giữa hai mức là $20 \log_{10}(127/8) \approx 24$ dB, đúng quy tắc quen thuộc "mỗi bit thêm vào cho khoảng 6 dB". Trường hợp outlier minh họa điểm yếu cố hữu của scale theo absmax: một outlier lớn gấp 20 lần làm $\Delta$ của cả block tăng theo, nên 31 phần tử còn lại bị lượng tử hóa thô hơn và SQNR giảm khoảng 5 đến 6 dB. Đây chính là động lực của K-quants (scale theo sub-block), I-quants với importance matrix, và các kỹ thuật xoay Hadamard được bàn ở các bài lý thuyết.

Cuối cùng, Q8_0 trên trọng số Gauss cho SQNR khoảng 45 dB, tức năng lượng nhiễu chỉ bằng khoảng $3 \times 10^{-5}$ năng lượng tín hiệu. Đó là lý do bảng của `llama-quantize` ghi Q8_0 chỉ tăng perplexity +0.0026 trên Llama-3-8B (Bài 8, Phần 8).

---

## Phần 6: Chạy inference từ Python

Cách đơn giản nhất là chạy `llama-server` (Bài 7) rồi gọi HTTP, không cần binding nào:

```python
import json, urllib.request

def complete(prompt: str, n_predict: int = 128) -> dict:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.7, "top_p": 0.95}
    req = urllib.request.Request("http://localhost:8080/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

out = complete("Giải thích quantization trong LLM inference:")
print(out["content"])
t = out["timings"]
print(f"prompt: {t['prompt_n']} tok @ {t['prompt_per_second']:.1f} tok/s, "
      f"gen: {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} tok/s")
```

Endpoint native `/completion` trả về văn bản trong trường `content` và thống kê thời gian trong `timings` (các trường `prompt_n`, `prompt_per_second`, `predicted_n`, `predicted_per_second` được tạo trong `tools/server/server-common.cpp`). Nếu muốn nhúng thư viện trực tiếp vào tiến trình Python, có thể dùng binding bên thứ ba `llama-cpp-python`; hãy lưu ý rằng nó đóng gói một phiên bản `libllama` riêng, có thể trễ hơn mainline vài tuần.

---

## 💡 Đúc kết Bài 9

Qua bài thực hành này, chúng ta đã:

1. **Tạo và đọc GGUF**: ghi file bằng `GGUFWriter`, rồi tự parse header, metadata, tensor info, tính đúng `data_offset` theo alignment và nạp dữ liệu, khớp hoàn toàn với `gguf.GGUFReader`.
2. **Cài đặt Q8_0 và Q4_0 bit-exact**: hiểu vì sao scale phải tính ở FP32, vì sao `roundf` khác `np.round`, vì sao Q4_0 dùng $d = m/(-8)$, và layout nibble của block.
3. **Đo chất lượng có lý thuyết đi kèm**: SQNR khoảng 45 dB cho Q8_0 và 21 dB cho Q4_0 trên dữ liệu Gauss, khớp với mô hình nhiễu $\Delta^2/12$, cùng với tác động của outlier.
4. **Gọi inference từ Python** qua HTTP API của `llama-server`.

Bài 9 khép lại phần lõi của giáo trình. Các bài tiếp theo đi sâu vào những hướng chuyên biệt như mô hình ternary (BitNet) và các họ model lượng tử hóa cực thấp.

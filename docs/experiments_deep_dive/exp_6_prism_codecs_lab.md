# Bài 6: Tự cài đặt các codec của Bonsai và kiểm chứng byte-exact với ggml

## Phần 1. Mục tiêu

Sau bài thực hành này, bạn sẽ:

- Tự viết bằng numpy bộ mã hóa và giải mã cho `Q1_0`, `Q2_0`, `PQ2_0`, `PTQ1_0` và `TQ1_0`.
- Kiểm chứng rằng mã của bạn cho ra **đúng từng byte** như thư viện C `libggml-base` của fork PrismML, gọi trực tiếp qua `ctypes`.
- Quan sát bằng số liệu ba hiện tượng lý thuyết đã học ở [Theory 9](../theory_deep_dive/theory_9_bonsai_gguf_formats) và [Theory 10](../theory_deep_dive/theory_10_hadamard_folding): định dạng ternary group 128 là lossless với trọng số đã ternary; `TQ1_0` mất mát khi hai nhóm 128 có scale khác nhau; phép xoay Hadamard làm giảm sai số ternary với phân phối đuôi dày.
- Kiểm chứng định lý fold $W x = W_f \mathbf{H} D x$ bằng số.

## Phần 2. Chuẩn bị

Yêu cầu: Python 3.10 trở lên với `numpy`, CMake, trình biên dịch C/C++. Không cần GPU.

```bash
git clone https://github.com/PrismML-Eng/llama.cpp prism-llama.cpp
cd prism-llama.cpp
cmake -B build-cpu -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-cpu --target ggml-base -j
ls build-cpu/bin/libggml-base.so      # macOS: libggml-base.dylib
cd ..
mkdir bonsai-lab && cd bonsai-lab
```

Chỉ cần target `ggml-base`: các hàm `ggml_quantize_chunk` và bảng `ggml_get_type_traits` (chứa con trỏ `to_float`) nằm trong `ggml/src/ggml.c` và `ggml/src/ggml-quants.c`, không phụ thuộc backend CPU.

## Phần 3. Bước 1: Cài đặt các codec

Tạo file `prism_codecs.py`. Đọc kỹ từng hàm và đối chiếu với struct trong `ggml/src/ggml-common.h`:

- `Q1_0`: scale absmean, bit $j$ của byte $\lfloor j/8 \rfloor$ là dấu của phần tử $j$ (`np.packbits` với `bitorder="little"`).
- `Q2_0`, `PQ2_0`: scale absmax, mã $q = \operatorname{clamp}(\operatorname{round}(w/d) + 1, 0, 3)$, phần tử $j$ ở byte $\lfloor j/4 \rfloor$, bit $2(j \bmod 4)$.
- `PTQ1_0`, `TQ1_0`: 5 trit mỗi byte theo các stage `{32, 16, 8}`, byte lưu $\lceil 256 v / 243 \rceil$, scale ở cuối block.

```python
"""Reference numpy codecs for the low-bit GGUF types used by the Bonsai models.

Q1_0   (ggml type 41, group 128): fp16 d, then 16 bytes of sign bits        -> 18 B / 128 w
Q2_0   (ggml type 42, group 64) : fp16 d, then 16 bytes of 2-bit codes       -> 18 B / 64 w
PQ2_0  (ggml type 142, group 128, Prism fork): fp16 d, then 32 bytes codes   -> 34 B / 128 w
PTQ1_0 (ggml type 143, group 128, Prism fork): 24 B qs + 2 B qh + fp16 d     -> 28 B / 128 w
TQ1_0  (ggml type 34, group 256): 48 B qs + 4 B qh + fp16 d                 -> 54 B / 256 w
"""
import numpy as np

F16 = np.float16


# ---------------------------------------------------------------- Q1_0 (sign bits)
def quantize_q1_0(x):
    x = x.reshape(-1, 128).astype(np.float32)
    d = np.abs(x).mean(axis=1).astype(F16)                      # absmean scale
    bits = (x >= 0).astype(np.uint8)                            # 1 -> +d, 0 -> -d
    qs = np.packbits(bits.reshape(-1, 16, 8), axis=-1, bitorder="little").reshape(-1, 16)
    out = np.zeros((x.shape[0], 18), np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs
    return out.tobytes()


def dequantize_q1_0(buf):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 18)
    d = b[:, 0:2].copy().view(F16).astype(np.float32)           # [nb, 1]
    bits = np.unpackbits(b[:, 2:], axis=-1, bitorder="little")  # [nb, 128]
    return ((2.0 * bits - 1.0) * d).reshape(-1)


# ---------------------------------------------------------------- Q2_0 / PQ2_0 (2-bit slots)
def _quantize_2bit(x, qk):
    x = x.reshape(-1, qk).astype(np.float32)
    d = np.abs(x).max(axis=1, keepdims=True)                    # absmax scale
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.round(x * inv).astype(np.int32) + 1, 0, 3).astype(np.uint8)  # {-1,0,1,2} -> {0,1,2,3}
    q = q.reshape(-1, qk // 4, 4)
    qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)  # element j at byte j/4, bits 2*(j%4)
    out = np.zeros((x.shape[0], 2 + qk // 4), np.uint8)
    out[:, 0:2] = d.astype(F16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs
    return out.tobytes()


def _dequantize_2bit(buf, qk):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 2 + qk // 4)
    d = b[:, 0:2].copy().view(F16).astype(np.float32)
    qs = b[:, 2:]
    q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, qk)
    return ((q.astype(np.float32) - 1.0) * d).reshape(-1)


quantize_q2_0 = lambda x: _quantize_2bit(x, 64)
dequantize_q2_0 = lambda b: _dequantize_2bit(b, 64)
quantize_pq2_0 = lambda x: _quantize_2bit(x, 128)
dequantize_pq2_0 = lambda b: _dequantize_2bit(b, 128)


# ---------------------------------------------------------------- base-3 packing (TQ1_0 / PTQ1_0)
def _stages(n_qs):
    """Byte groups of the qs array: TQ1_0 (48 B) -> [32, 16], PTQ1_0 (24 B) -> [16, 8]."""
    out, j = [], 0
    for c in (32, 16, 8):
        while j + c <= n_qs:
            out.append(c)
            j += c
    return out


def _pack5(t):
    """t: [5, c] trits in {0,1,2}; most significant trit first; returns ceil(v * 256 / 243)."""
    v = np.zeros(t.shape[1], np.int32)
    for n in range(5):
        v = v * 3 + t[n]
    return ((v * 256 + 242) // 243).astype(np.uint8)


def _unpack(byte, ntrits):
    out = []
    for n in range(ntrits):
        q = (byte.astype(np.uint16) * (3 ** n)) & 0xFF          # uint8 wrap-around keeps the fraction
        out.append(((q * 3) >> 8).astype(np.int32))             # top trit of the remaining fraction
    return out


def _quantize_trits(x, qk):
    n_qh = qk // 64
    n_qs = (qk - 4 * n_qh) // 5
    x = x.reshape(-1, qk).astype(np.float32)
    blocks = []
    for row in x:
        d = np.abs(row).max()
        inv = 1.0 / d if d else 0.0
        t = (np.round(row * inv).astype(np.int32) + 1)          # {0,1,2}
        qs, pos = [], 0
        for c in _stages(n_qs):
            qs.append(_pack5(t[pos:pos + 5 * c].reshape(5, c)))
            pos += 5 * c
        th = t[pos:pos + 4 * n_qh].reshape(4, n_qh)
        v = np.zeros(n_qh, np.int32)
        for m in range(4):
            v = v * 3 + th[m]
        v = v * 3                                                # 4 trits, shifted to the top
        qh = ((v * 256 + 242) // 243).astype(np.uint8)
        blocks.append(np.concatenate(qs).tobytes() + qh.tobytes() + np.array([d], F16).tobytes())
    return b"".join(blocks)


def _dequantize_trits(buf, qk):
    n_qh = qk // 64
    n_qs = (qk - 4 * n_qh) // 5
    bs = n_qs + n_qh + 2
    b = np.frombuffer(buf, np.uint8).reshape(-1, bs)
    out = []
    for blk in b:
        d = np.float32(blk[-2:].copy().view(F16)[0])
        vals, j = [], 0
        for c in _stages(n_qs):
            for tr in _unpack(blk[j:j + c], 5):
                vals.append(tr)
            j += c
        for tr in _unpack(blk[n_qs:n_qs + n_qh], 4):
            vals.append(tr)
        out.append((np.concatenate(vals) - 1).astype(np.float32) * d)
    return np.concatenate(out)


quantize_ptq1_0 = lambda x: _quantize_trits(x, 128)
dequantize_ptq1_0 = lambda b: _dequantize_trits(b, 128)
quantize_tq1_0 = lambda x: _quantize_trits(x, 256)
dequantize_tq1_0 = lambda b: _dequantize_trits(b, 256)


# ---------------------------------------------------------------- Hadamard fold
def sylvester(n):
    i = np.arange(n)
    parity = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(parity, -1.0, 1.0) / np.sqrt(n)              # same construction as load_tensors_hadamard


def fwht(x):
    """Normalized fast Walsh-Hadamard transform along the last axis (length power of two)."""
    x = x.astype(np.float64).copy()
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = x[..., 0, :].copy(), x[..., 1, :].copy()
        x[..., 0, :], x[..., 1, :] = a + b, a - b
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x / np.sqrt(n)


def block_transform(x, B, signs):
    """Activation-side transform of llama.cpp: x -> H_B (s * x), blockwise along the last axis."""
    y = x * signs
    y = y.reshape(*y.shape[:-1], -1, B)
    return fwht(y).reshape(x.shape)
```

## Phần 4. Bước 2: Đối chiếu với thư viện C

Tạo `verify_against_ggml.py`. Script lượng tử ba loại tensor ($8 \times 1024$): trọng số **ternary chính xác** ở group 128 (như một checkpoint Bonsai xuất ra), trọng số **nhị phân chính xác** $\pm s_g$, và trọng số **Gauss** thông thường. Với mỗi kiểu, nó so sánh byte của C và Python, so sánh kết quả giải nén, và in sai số tương đối $\lVert \hat{w} - w \rVert / \lVert w \rVert$.

```python
"""Cross-check prism_codecs.py against the C reference in libggml-base (Prism fork build)."""
import ctypes, sys
import numpy as np
sys.path.insert(0, sys.argv[2])
import prism_codecs as pc

lib = ctypes.CDLL(sys.argv[1])


class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)

TYPES = {"q1_0": 41, "q2_0": 42, "pq2_0": 142, "ptq1_0": 143, "tq1_0": 34}
rng = np.random.default_rng(0)
n_per_row, nrows = 1024, 8

# exact ternary weights at group 128 (what a Bonsai export contains), plus a random tensor
scales = rng.uniform(0.01, 0.05, size=(nrows, n_per_row // 128)).astype(np.float16).astype(np.float32)
trits = rng.integers(-1, 2, size=(nrows, n_per_row)).astype(np.float32)
ternary = (trits.reshape(nrows, -1, 128) * scales[..., None]).reshape(nrows, n_per_row)
signs = np.where(rng.random((nrows, n_per_row)) < 0.5, -1.0, 1.0)
binary = (signs.reshape(nrows, -1, 128) * scales[..., None]).reshape(nrows, n_per_row).astype(np.float32)
gauss = rng.standard_normal((nrows, n_per_row)).astype(np.float32) * 0.02

for name, t in TYPES.items():
    tr = lib.ggml_get_type_traits(t).contents
    for label, x in (("ternary", ternary), ("binary", binary), ("gauss", gauss)):
        x = np.ascontiguousarray(x, np.float32)
        nbytes = nrows * n_per_row // tr.blck_size * tr.type_size
        dst = (ctypes.c_uint8 * nbytes)()
        lib.ggml_quantize_chunk(t, x.ctypes.data, dst, 0, nrows, n_per_row, None)
        c_bytes = bytes(dst)
        py_bytes = getattr(pc, "quantize_" + name)(x)
        y_c = np.zeros(x.size, np.float32)
        TO_FLOAT(tr.to_float)(ctypes.cast(dst, ctypes.c_void_p), y_c.ctypes.data, x.size)
        y_py = getattr(pc, "dequantize_" + name)(c_bytes)
        rel = np.linalg.norm(y_c - x.ravel()) / np.linalg.norm(x)
        print(f"{name:7s} {label:8s} bpw={8*tr.type_size/tr.blck_size:.4f} bytes_equal={c_bytes == py_bytes} "
              f"dequant_equal={np.array_equal(y_c, y_py)} rel_err={rel:.5f}")

# Hadamard fold: W x == W_f (H (s * x)) with W_f = W D H, blockwise
B, n_in, n_out = 128, 512, 64
W = rng.standard_normal((n_out, n_in))
s = np.where(rng.random(n_in) < 0.5, -1.0, 1.0)
H = pc.sylvester(B)
Hblk = np.kron(np.eye(n_in // B), H)
W_f = W @ np.diag(s) @ Hblk
x = rng.standard_normal(n_in)
print("H symmetric & orthogonal:", np.allclose(H, H.T), np.allclose(H @ H, np.eye(B)))
print("fwht == dense H:", np.allclose(pc.fwht(x[:B]), H @ x[:B]))
print("fold exact:", np.allclose(W @ x, W_f @ pc.block_transform(x, B, s)))
```

Chạy (luôn dùng `-I` để Python không nạp module lạ từ thư mục hiện tại):

```bash
python3 -I verify_against_ggml.py ../prism-llama.cpp/build-cpu/bin/libggml-base.so .
```

Kết quả mong đợi (đã chạy trên fork nhánh `prism`, commit `73bfe6c`):

```text
q1_0    ternary  bpw=1.1250 bytes_equal=True dequant_equal=True rel_err=0.57531
q1_0    binary   bpw=1.1250 bytes_equal=True dequant_equal=True rel_err=0.00000
q1_0    gauss    bpw=1.1250 bytes_equal=True dequant_equal=True rel_err=0.59622
q2_0    ternary  bpw=2.2500 bytes_equal=True dequant_equal=True rel_err=0.00000
q2_0    binary   bpw=2.2500 bytes_equal=True dequant_equal=True rel_err=0.00000
q2_0    gauss    bpw=2.2500 bytes_equal=True dequant_equal=True rel_err=0.73236
pq2_0   ternary  bpw=2.1250 bytes_equal=True dequant_equal=True rel_err=0.00000
pq2_0   binary   bpw=2.1250 bytes_equal=True dequant_equal=True rel_err=0.00000
pq2_0   gauss    bpw=2.1250 bytes_equal=True dequant_equal=True rel_err=0.77865
ptq1_0  ternary  bpw=1.7500 bytes_equal=True dequant_equal=True rel_err=0.00000
ptq1_0  binary   bpw=1.7500 bytes_equal=True dequant_equal=True rel_err=0.00000
ptq1_0  gauss    bpw=1.7500 bytes_equal=True dequant_equal=True rel_err=0.77865
tq1_0   ternary  bpw=1.6875 bytes_equal=True dequant_equal=True rel_err=0.26245
tq1_0   binary   bpw=1.6875 bytes_equal=True dequant_equal=True rel_err=0.26740
tq1_0   gauss    bpw=1.6875 bytes_equal=True dequant_equal=True rel_err=0.81959
H symmetric & orthogonal: True True
fwht == dense H: True
fold exact: True
```

## Phần 5. Bước 3: Đọc kết quả

**Tính đúng của cài đặt.** Mọi dòng phải có `bytes_equal=True` và `dequant_equal=True`. Nếu một dòng sai, lỗi thường gặp nhất là đặt scale sai vị trí (đầu block với `Q*_0`, cuối block với `*TQ1_0`) hoặc xếp mã 2 bit theo kiểu "4 luồng cách nhau 32" của `TQ2_0` thay vì tuần tự.

**Lossless với trọng số đã ternary.** `PQ2_0`, `Q2_0` và `PTQ1_0` đều cho `rel_err=0.00000` trên dữ liệu ternary và nhị phân: đúng như mệnh đề ở Theory 9, Phần 4.

**`TQ1_0` mất mát.** Cùng dữ liệu ternary group 128, `TQ1_0` (block 256) cho sai số khoảng 26%, vì mỗi block 256 chỉ giữ scale lớn hơn của hai nhóm. Con số phụ thuộc mạnh vào độ chênh scale: ở đây scale được rút đều trên $[0.01, 0.05]$; trên một checkpoint thật với các scale liền kề gần nhau, fork báo cáo khoảng 6.43%.

**Bộ lượng tử tham chiếu không dành cho mô hình thường.** Trên dữ liệu Gauss, `PQ2_0` và `PTQ1_0` cho sai số khoảng 0.78, tệ hơn nhiều so với khoảng 0.43 của bộ ternary tối ưu (bước 4). `Q1_0` cho khoảng 0.596, sát giá trị lý thuyết $\sqrt{1 - 2/\pi} \approx 0.603$.

**Fold Hadamard.** Ba dòng cuối xác nhận $H_B$ đối xứng, trực giao, FWHT trùng với nhân ma trận dày đặc, và $W x = W_f \mathbf{H} (D x)$.

## Phần 6. Bước 4: Phép xoay giảm sai số ternary như thế nào

Tạo `rotation_effect.py`. Script dùng một bộ lượng tử ternary **tối ưu** theo bình phương tối thiểu cho từng nhóm 128 (với mỗi ngưỡng $k$ phần tử lớn nhất, scale tối ưu là trung bình trị tuyệt đối của chúng; chọn $k$ cực đại hóa $(\sum_{i \le k} |w|_{(i)})^2 / k$), rồi so sánh sai số khi lượng tử trực tiếp và khi lượng tử trong cơ sở xoay (block 512) rồi xoay ngược.

```python
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np, prism_codecs as pc
rng = np.random.default_rng(1)

def ternary_ls(w, G=128):
    """Best-of-threshold least-squares ternary quantizer per group (oracle, not Prism's method)."""
    w = w.reshape(-1, G); out = np.empty_like(w)
    for i, g in enumerate(w):
        a = np.sort(np.abs(g))[::-1]; c = np.cumsum(a); k = np.arange(1, G + 1)
        k_best = np.argmax(c**2 / k)                     # maximize (sum|w|)^2/k  <=> minimize residual
        s = c[k_best] / (k_best + 1); thr = a[k_best]
        out[i] = np.where(np.abs(g) >= thr, np.sign(g) * s, 0.0)
    return out.reshape(-1)

def kurt(x): x = x - x.mean(); return (x**4).mean() / (x**2).mean()**2

n_out, n_in, B = 256, 2048, 512
for name, W in (("gauss", rng.standard_normal((n_out, n_in))),
                ("student-t(3)", rng.standard_t(3, (n_out, n_in))),
                ("gauss+1%outlier-ch", rng.standard_normal((n_out, n_in)) * np.where(rng.random(n_in) < 0.01, 20, 1))):
    s = np.where(rng.random(n_in) < 0.5, -1.0, 1.0)
    Wr = pc.block_transform(W, B, s)                       # rotated basis W D H (rows transformed)
    e0 = np.linalg.norm(ternary_ls(W.ravel()) - W.ravel()) / np.linalg.norm(W)
    q = ternary_ls(Wr.ravel()).reshape(W.shape)
    # map back to the original basis: (W D H) H D = W
    back = pc.block_transform(q, B, np.ones(n_in)) * s
    e1 = np.linalg.norm(back - W) / np.linalg.norm(W)
    print(f"{name:20s} kurtosis {kurt(W.ravel()):7.2f} -> {kurt(Wr.ravel()):5.2f} | ternary rel err {e0:.4f} -> {e1:.4f}")
```

```bash
python3 -I rotation_effect.py .
```

Kết quả mong đợi:

```text
gauss                kurtosis    3.02 ->  3.00 | ternary rel err 0.4335 -> 0.4323
student-t(3)         kurtosis  330.55 ->  3.63 | ternary rel err 0.5596 -> 0.4216
gauss+1%outlier-ch   kurtosis  187.68 ->  3.23 | ternary rel err 0.4169 -> 0.4016
```

Đọc kết quả: với phân phối Gauss, phép xoay không mang lại gì (phân phối đã "tròn"). Với Student-t bậc 3 (đuôi rất dày), kurtosis giảm từ hàng trăm về gần 3 và sai số giảm khoảng 25%. Với outlier tập trung ở 1% kênh đầu vào, mức giảm nhỏ hơn vì chuẩn của sai số bị chi phối bởi chính các kênh lớn đó.

## Phần 7. Bài tập mở rộng

1. Viết `vec_dot_pq2_0_q8_0` bằng numpy theo công thức gộp độ lệch: $d_w d_x (\sum c_i a_i - \sum a_i)$, và kiểm tra với tích vô hướng của hai vector đã giải nén.
2. Sửa `_pack5` để dùng làm tròn xuống thay vì lên. Tìm tất cả $v \in [0, 242]$ bị giải mã sai và đối chiếu với chứng minh ở Theory 9, Phần 5.2.
3. Thay ma trận dày đặc bằng `fwht` trong kiểm tra fold, đo thời gian với $B \in \{512, 1024, 4096\}$ và so với dự đoán $B^2$ so với $B \log_2 B$.
4. Đọc metadata của một file GGUF Bonsai 2 bằng `gguf-dump` (gói `gguf` của llama.cpp) và liệt kê các key `prism.hadamard.*`. Kiểm tra rằng `block_size` chia hết chiều đầu vào của mọi tensor trong `weight_names`.

## Phần 8. Tiêu chí hoàn thành

- [ ] Tất cả 15 dòng của bước 2 có `bytes_equal=True dequant_equal=True`.
- [ ] Giải thích được vì sao `TQ1_0` mất mát còn `PTQ1_0` thì không, trên cùng dữ liệu.
- [ ] Giải thích được vì sao tự lượng tử một mô hình FP16 thường sang `PQ2_0` cho kết quả vô nghĩa.
- [ ] Tái tạo được bảng của bước 4 và giải thích vai trò của kurtosis.

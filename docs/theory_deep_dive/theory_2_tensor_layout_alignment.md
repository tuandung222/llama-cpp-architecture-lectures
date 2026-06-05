---
sidebar_position: 3
sidebar_label: "Bài 2: Đại số Tensor Layout & Memory Alignment"
---

# Bài 2: Đại số Tensor Layout & Memory Alignment

Bài này phân tích toán học của tensor memory layout trong GGML, từ stride computation đến SIMD alignment requirements.

---

## 1. Stride Computation trong GGML

Với `ggml_tensor` có dimensions `ne[4]` và type $T$:

$$nb[0] = \text{type\_size}(T)$$
$$nb[1] = nb[0] \times \frac{ne[0]}{\text{block\_size}(T)} + \text{padding}$$
$$nb[i] = nb[i-1] \times ne[i-1] \quad \text{for } i \geq 2$$

**Ví dụ**: Tensor F32 có shape [4096, 4096]:
- $nb[0] = 4$ bytes (sizeof(float))
- $nb[1] = 4 \times 4096 = 16384$ bytes
- $nb[2] = 16384 \times 4096 = 67,108,864$ bytes

---

## 2. SIMD Alignment: Tại sao 32 bytes?

AVX2 sử dụng thanh ghi 256-bit = 32 bytes. Aligned load (`_mm256_load_si256`) yêu cầu:

$$\text{address} \equiv 0 \pmod{32}$$

Với block quantization Q4_0 (18 bytes/block), để đảm bảo alignment:

$$\text{tensor\_data\_offset} \equiv 0 \pmod{32}$$

Đây là lý do `GGUF_DEFAULT_ALIGNMENT = 32`.

---

## 3. Block Packing Algebra

| Quant | Block size | Bytes/block | Packed layout |
|:---|:---|:---|:---|
| Q4_0 | 32 | 18 | [d:2][qs:16] |
| Q4_1 | 32 | 20 | [d:2][m:2][qs:16] |
| Q4_K | 256 | 144 | [d:2][dmin:2][scales:12][qs:128] |
| Q8_0 | 32 | 34 | [d:2][qs:32] |

Hiệu suất packing:

$$\text{Efficiency} = \frac{\text{Data bits}}{\text{Total bits}} = \frac{n \times b}{\text{block\_bytes} \times 8}$$

Q4_0: $\frac{32 \times 4}{18 \times 8} = \frac{128}{144} = 88.9\%$ (11.1% overhead cho scale).

---

## 4. Contiguity Conditions

Tensor được gọi là **contiguous** (liên tục) khi:

$$nb[i] = nb[i-1] \times ne[i-1] \quad \forall i$$

Tensor transposed (hoán vị chiều 0 và 1) KHÔNG contiguous, nhưng GGML vẫn xử lý được nhờ stride-based access.

---

## 💡 Đúc kết

Tensor layout trong GGML được thiết kế để tối ưu cho SIMD: 32-byte alignment đảm bảo vectorized loads đạt peak bandwidth, block packing tối thiểu overhead, và stride-based access cho phép xử lý transposed tensors mà không cần sao chép dữ liệu.

---
sidebar_position: 3
sidebar_label: "Bài 2: Đại số Tensor Layout & Memory Alignment"
---

# Bài 2: Đại số Tensor Layout & Memory Alignment

Bài này phân tích toán học của tensor memory layout trong GGML, từ stride computation, điều kiện contiguous, đến các tầng alignment khác nhau (arena, backend buffer, file GGUF) và những gì alignment thực sự đảm bảo cho kernel SIMD.

---

## 1. Stride Computation trong GGML

Với `ggml_tensor` có dimensions `ne[4]` và type $T$ có block size $B_T$ (`ggml_blck_size`, bằng 1 với F32/F16) và kích thước block $S_T$ byte (`ggml_type_size`):

$$nb[0] = S_T$$
$$nb[1] = nb[0] \times \frac{ne[0]}{B_T}$$
$$nb[i] = nb[i-1] \times ne[i-1] \quad \text{for } i \geq 2$$

Địa chỉ byte của phần tử (hoặc block) có chỉ số $(i_0, i_1, i_2, i_3)$ là

$$\text{addr} = \text{data} + \frac{i_0}{B_T}\, nb[0] + i_1\, nb[1] + i_2\, nb[2] + i_3\, nb[3]$$

với $i_0$ là bội số của $B_T$ khi $T$ là kiểu quantized: không thể định địa chỉ một phần tử riêng lẻ bên trong block, chỉ có thể định địa chỉ cả block. Đây là lý do mọi op trên tensor quantized làm việc theo **hàng** (`ne[0]` phần tử) và yêu cầu `ne[0] % B_T == 0` (kiểm tra này nằm trong `gguf.cpp` khi đọc file).

> **Lưu ý**: Số hạng `padding` trong comment của `ggml.h` chỉ xuất hiện ở **view**: khi một tensor là view vào tensor lớn hơn (ví dụ `get_k()` trả về view của $n_{kv}$ ô đầu tiên trong KV Cache), stride `nb[1]`, `nb[2]` được kế thừa từ tensor gốc và có thể lớn hơn kích thước thực của hàng.

**Ví dụ 1**: Tensor F32 có shape [4096, 4096]:
- $nb[0] = 4$ bytes (sizeof(float))
- $nb[1] = 4 \times 4096 = 16384$ bytes
- $nb[2] = 16384 \times 4096 = 67{,}108{,}864$ bytes

**Ví dụ 2**: Cùng shape nhưng kiểu Q4_K ($B_T = 256$, $S_T = 144$):
- $nb[0] = 144$ bytes (một super-block)
- $nb[1] = 144 \times 4096/256 = 2304$ bytes
- $nb[2] = 2304 \times 4096 = 9{,}437{,}184$ bytes $= 9$ MiB, đúng $4.5$ bit/trọng số.

**Ví dụ 3**: Transpose bằng `ggml_transpose()` chỉ hoán đổi $(ne[0], ne[1])$ và $(nb[0], nb[1])$. Với F32 [4096, 4096] ta được $nb[0] = 16384$, $nb[1] = 4$: phần tử kề nhau theo chiều 0 cách nhau 16 KB, một layout rất tệ cho SIMD. Vì vậy trong llama.cpp, sau `ggml_permute`/`ggml_transpose` thường có `ggml_cont()` (sao chép thành contiguous) trước các op cần đọc liên tục.

---

## 2. Contiguity Conditions

Tensor được gọi là **contiguous** khi stride khớp với công thức ở Phần 1. Định nghĩa thực tế trong `ggml_is_contiguous_m_n()` (`ggml/src/ggml.c`) tinh tế hơn một đẳng thức đơn giản:

$$nb[0] = S_T \;\text{(trừ khi } ne[0] = B_T\text{)}, \qquad nb[i] = S_T \cdot \frac{ne[0]}{B_T} \prod_{k=1}^{i-1} ne[k] \;\;\text{với mọi } i \text{ mà } ne[i] \neq 1$$

Hai ngoại lệ cần chú ý: chiều có $ne[i] = 1$ được **bỏ qua** (stride của nó không quan trọng vì chỉ số luôn bằng 0), và có các biến thể `ggml_is_contiguous_1`, `ggml_is_contiguous_2` cho phép các chiều thấp không liên tục, chỉ yêu cầu các chiều cao hơn liên tục ("contiguous theo hàng"). Rất nhiều kernel chỉ cần `ggml_is_contiguous_1`: mỗi hàng liên tục là đủ, còn các hàng có thể cách nhau tùy ý.

Tensor transposed (hoán vị chiều 0 và 1) KHÔNG contiguous, nhưng GGML vẫn xử lý được nhờ stride-based access trong các op hỗ trợ nó.

---

## 3. Ba tầng alignment trong GGML

| Tầng | Hằng số | Giá trị | Nơi định nghĩa | Đảm bảo điều gì |
|:---|:---|:---|:---|:---|
| Arena `ggml_context` | `GGML_MEM_ALIGN` | 16 (64-bit), 8 (wasm64), 4 (32-bit) | `ggml/include/ggml.h` | Mỗi `ggml_object` và struct `ggml_tensor` trong arena |
| Backend buffer | `TENSOR_ALIGNMENT` | 32 | `ggml/src/ggml-impl.h` | `tensor->data` của mọi tensor do buffer CPU cấp phát (`ggml_backend_cpu_buffer_type_get_alignment`) |
| File GGUF | `GGUF_DEFAULT_ALIGNMENT` (hoặc `general.alignment`) | 32 | `ggml/include/gguf.h` | Offset bắt đầu của data blob và của mỗi tensor trong file |

Tầng thứ ba có điều kiện chặt: `general.alignment` phải là `UINT32` và là lũy thừa của 2, vì phép làm tròn dùng macro bitwise

$$\text{GGML\_PAD}(x, A) = (x + A - 1) \;\&\; \lnot (A - 1)$$

chỉ đúng khi $A = 2^k$. Đọc công thức: cộng $A - 1$ để vượt qua biên kế tiếp (trừ khi $x$ đã nằm đúng biên), rồi xóa $k$ bit thấp để lùi về biên đó.

---

## 4. SIMD Alignment: 32 byte đảm bảo được gì, và không đảm bảo gì

AVX2 sử dụng thanh ghi 256-bit = 32 bytes. Lệnh aligned load (`_mm256_load_si256`) yêu cầu:

$$\text{address} \equiv 0 \pmod{32}$$

GGUF đảm bảo $\text{tensor\_data\_offset} \equiv 0 \pmod{32}$. Khi file được `mmap` (ánh xạ bắt đầu ở biên trang, $4096 \equiv 0 \pmod{32}$), con trỏ `tensor->data` trỏ thẳng vào vùng map cũng căn lề 32 byte.

Nhưng xét các block **bên trong** tensor Q4_0 (18 bytes/block): block thứ $k$ nằm tại offset $18k$, và

$$18k \equiv 0 \pmod{32} \iff 16 \mid k$$

nên chỉ 1 trên 16 block được căn lề 32 byte. Tổng quát, block thứ $k$ căn lề khi $\frac{32}{\gcd(S_T, 32)} \mid k$. Với Q4_K ($S_T = 144 = 16 \cdot 9$), $\gcd = 16$, nên chỉ block chẵn căn lề. Đó là lý do các kernel trong `ggml/src/ggml-cpu/arch/x86/quants.c` dùng **unaligned load** (`_mm256_loadu_si256`, `_mm_loadu_si128`). Trên x86 hiện đại, unaligned load không vắt qua biên cache line 64 byte có chi phí gần như bằng aligned load, nên thiết kế này không mất hiệu năng đáng kể.

Khi bố cục block thật sự cản trở kernel, GGML **đổi bố cục lúc runtime** thay vì đổi định dạng file. `ggml/src/ggml-cpu/repack.h` định nghĩa template

```cpp
template <int K, int N> struct block {
    ggml_half d[N];                         // N scale của N block gốc
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants của N block, xen kẽ
};
using block_q4_0x4 = block<4, 4>;   // 4 block Q4_0 gộp: 4 x 2 + 64 = 72 bytes
using block_q4_0x8 = block<4, 8>;   // 8 block Q4_0 gộp: 8 x 2 + 128 = 144 bytes
```

Gộp $N$ block của $N$ **hàng khác nhau** và xen kẽ quants giúp kernel GEMM tính $N$ hàng kết quả song song, đồng thời tách các scale ra một vùng liên tục. Đây chính là thay thế cho các `ggml_type` cũ `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8` đã bị xóa khỏi file GGUF.

---

## 5. Block Packing Algebra

| Quant | Block size | Bytes/block | Packed layout (theo thứ tự field trong `ggml-common.h`) |
|:---|:---|:---|:---|
| Q1_0 | 128 | 18 | [d:2][qs:16] |
| Q2_0 | 64 | 18 | [d:2][qs:16] |
| Q4_0 | 32 | 18 | [d:2][qs:16] |
| Q4_1 | 32 | 20 | [d:2][m:2][qs:16] |
| Q5_0 | 32 | 22 | [d:2][qh:4][qs:16] |
| Q8_0 | 32 | 34 | [d:2][qs:32] |
| MXFP4 | 32 | 17 | [e:1][qs:16] |
| NVFP4 | 64 | 36 | [d:4][qs:32] |
| Q4_K | 256 | 144 | [d:2][dmin:2][scales:12][qs:128] |
| Q6_K | 256 | 210 | [ql:128][qh:64][scales:16][d:2] |
| TQ2_0 | 256 | 66 | [qs:64][d:2] |
| Q8_K | 256 | 292 | [d:4 (float)][qs:256][bsums:32] |

Hiệu suất packing:

$$\text{Efficiency} = \frac{\text{Data bits}}{\text{Total bits}} = \frac{n \times b}{\text{block\_bytes} \times 8}$$

với $n$ là block size và $b$ là số bit của mã.

- Q4_0: $\frac{32 \times 4}{18 \times 8} = \frac{128}{144} = 88.9\%$ (11.1% overhead cho scale).
- Q4_K: $\frac{256 \times 4}{144 \times 8} = \frac{1024}{1152} = 88.9\%$, cùng overhead với Q4_0 nhưng overhead đó mua được cả scale lẫn min cho mỗi sub-block 32.
- Q8_0: $\frac{32 \times 8}{34 \times 8} = 94.1\%$.
- MXFP4: $\frac{32 \times 4}{17 \times 8} = 94.1\%$, nhờ scale E8M0 chỉ 1 byte.
- Q1_0: $\frac{128 \times 1}{18 \times 8} = 88.9\%$: block 128 được chọn chính là để giữ overhead ở mức này khi mỗi mã chỉ còn 1 bit.

Mẫu hình chung: **overhead tương đối = bit metadata / (n x b)**. Khi giảm $b$, muốn giữ overhead cố định thì phải tăng $n$ (block lớn hơn) hoặc nén chính metadata (scale 6-bit của K-quants, scale E8M0 của MXFP4).

---

## 💡 Đúc kết

Tensor layout trong GGML là đại số stride đơn giản nhưng có hệ quả sâu: tensor quantized chỉ định địa chỉ được theo block, nên mọi op làm việc theo hàng. Alignment 32 byte của GGUF và của backend buffer đảm bảo **đầu tensor** căn lề (quan trọng cho `mmap` zero-copy và DMA), còn các block bên trong thường không căn lề, nên kernel dùng unaligned load; khi cần bố cục tốt hơn, backend CPU repack dữ liệu lúc nạp. Hiệu suất packing được điều khiển bởi tỷ lệ giữa bit metadata và kích thước block.

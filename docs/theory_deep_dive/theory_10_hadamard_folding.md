---
sidebar_position: 11
sidebar_label: "Bài 10: Fold Hadamard - Trọng số trong cơ sở xoay và runtime FWHT"
---

# Bài 10: Fold Hadamard: trọng số trong cơ sở xoay và runtime FWHT của llama.cpp

Bonsai 2 lưu trọng số ternary trong một cơ sở đã xoay bởi một ma trận Walsh-Hadamard có dấu. Điều đó biến một bài toán lượng tử hóa thành một bài toán **hợp đồng giữa file và runtime**: file chứa $W_f$, và runtime có nghĩa vụ biến đổi activation đúng cách trước mỗi phép nhân ma trận có fold. Bài này trình bày toán học của phép fold, thuật toán FWHT, và cách llama.cpp (mainline và fork PrismML) hiện thực hợp đồng đó, đến mức tên hàm và từng phép kiểm tra.

---

## Phần 1: Vì sao xoay trước khi lượng tử

Một bộ lượng tử đều với rất ít mức phải chọn một scale cho cả nhóm. Nếu nhóm chứa một vài giá trị rất lớn (outlier), hoặc scale bị kéo lên làm các giá trị nhỏ rơi hết về 0, hoặc outlier bị cắt. Cả hai đều tốn sai số. Độ "đuôi dày" được đo bằng kurtosis $\kappa = \mathbb{E}[(w-\mu)^4] / \sigma^4$; phân phối Gauss có $\kappa = 3$.

Nhân một vector $w \in \mathbb{R}^n$ với một ma trận trực giao "trộn đều" như Hadamard chuẩn hóa $H_n$ cho ra $u = H_n w$ với mỗi thành phần $u_j = \frac{1}{\sqrt{n}}\sum_i \pm w_i$. Theo định lý giới hạn trung tâm, mỗi $u_j$ là tổng có dấu của nhiều thành phần, nên phân phối của $u$ tiến gần Gauss và các outlier bị "trải mỏng" ra toàn bộ block. Thí nghiệm ở [Lab 6](../experiments_deep_dive/exp_6_prism_codecs_lab) với trọng số Student-t bậc 3: kurtosis giảm từ khoảng 330 xuống 3.6, sai số ternary tối ưu giảm từ 0.560 xuống 0.422. Dấu ngẫu nhiên $S$ được thêm vào để phá vỡ mọi cấu trúc tương quan giữa vector và các hàng cố định của $H_n$.

Cùng nguyên lý này cũng được mainline llama.cpp dùng ở một chỗ khác, độc lập với Bonsai: khi KV Cache bị lượng tử, `llama_kv_cache` tự xoay Q, K, V bằng một ma trận Hadamard sinh bởi `ggml_gen_hadamard` (`src/llama-kv-cache.cpp`), có thể tắt bằng biến môi trường `LLAMA_ATTN_ROT_DISABLE` (xem [Theory 4](theory_4_kv_cache_quantization)). Điểm khác biệt: ở KV Cache phép xoay được áp dụng và hoàn nguyên hoàn toàn trong runtime, còn ở Bonsai 2 phép xoay được fold vào trọng số lưu trên đĩa.

---

## Phần 2: Toán học của phép fold

### 2.1. Định nghĩa

Gọi $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ là trọng số trong cơ sở gốc, $D = \operatorname{diag}(s)$ với $s \in \{-1,+1\}^{d_{\text{in}}}$, và $\mathbf{H} = I_{d_{\text{in}}/B} \otimes H_B$ là ma trận Hadamard chuẩn hóa theo block kích thước $B$. Trọng số được lưu (folded weight) là:

$$W_f = W\, D\, \mathbf{H}$$

Đây đúng là quy ước trong comment của `load_hparams_hadamard` (`src/llama-model.cpp`): "a folded weight `W_f = W*D*H` is the weight W with the +1/-1 signs D and the normalized block Hadamard H folded in".

### 2.2. Định lý tương đương

**Mệnh đề.** Với mọi $x \in \mathbb{R}^{d_{\text{in}}}$: $W x = W_f\, \mathbf{H}\, (D x)$.

**Chứng minh.** Ta cần hai tính chất: $D^2 = I$ (hiển nhiên vì $s_i^2 = 1$) và $\mathbf{H}^2 = I$. Khi đó:

$$W_f\, \mathbf{H}\, D x = W D \mathbf{H} \mathbf{H} D x = W D D x = W x. \qquad \blacksquare$$

Tính chất $\mathbf{H}^2 = I$ đến từ việc $H_B$ vừa đối xứng vừa trực giao. Ma trận Sylvester được định nghĩa đệ quy:

$$H_1 = [1], \qquad H_{2n} = \frac{1}{\sqrt{2}}\begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix} = H_2 \otimes H_n$$

Với $H_2 = \frac{1}{\sqrt 2}\begin{pmatrix}1 & 1\\1 & -1\end{pmatrix}$ ta có $H_2^\top = H_2$ và $H_2^2 = I$. Tích Kronecker bảo toàn cả hai tính chất: $(A \otimes B)^\top = A^\top \otimes B^\top$ và $(A\otimes B)(C \otimes D) = AC \otimes BD$. Quy nạp cho $H_B^\top = H_B$, $H_B^2 = I$ với mọi $B$ là lũy thừa của 2. Phần tử tường minh là:

$$(H_B)_{r,c} = \frac{(-1)^{\operatorname{popcount}(r \,\&\, c)}}{\sqrt{B}}$$

và đây chính xác là vòng lặp tạo ma trận trong `llama_model_base::load_tensors_hadamard`: tính parity của `row & col` bằng các phép XOR dịch bit, gán $\pm 1/\sqrt{B}$.

Đọc định lý theo nghĩa thực tế: runtime **không bao giờ** khôi phục $W$. Nó giữ $W_f$ (ternary, nhỏ) và thay vào đó biến đổi activation: nhân dấu, rồi FWHT theo block. Ternary hóa được thực hiện trên $W_f$, tức trên trọng số đã xoay.

### 2.3. Hệ quả cho fine-tuning

Vì phép fold khả nghịch, có thể "mở" (unfold) mô hình về cơ sở gốc: $W = W_f\, \mathbf{H}\, D$ (vì $W_f \mathbf{H} D = W D \mathbf{H}\mathbf{H} D = W$). Kết quả là một checkpoint 16 bit chuẩn, nạp được bởi `transformers` mà không cần sửa mã, nhưng **mất cấu trúc ternary**: $W$ không còn là ternary nhân scale. Đây là con đường để fine-tune Bonsai 2 bằng các framework thông thường, với cái giá là bộ nhớ của mô hình 16 bit.

---

## Phần 3: Fast Walsh-Hadamard Transform

Nhân trực tiếp với $H_B$ tốn $B^2$ phép nhân-cộng mỗi block. Cấu trúc Kronecker cho phép phân rã thành $\log_2 B$ tầng "butterfly", mỗi tầng $B$ phép cộng trừ:

```text
FWHT(x[0..B-1]):                 # x đã được nhân với 1/sqrt(B)
    h = 1
    while h < B:
        for i in 0, 2h, 4h, ...:
            for j in i .. i+h-1:
                a, b = x[j], x[j+h]
                x[j], x[j+h] = a + b, a - b
        h = 2h
```

| $B$ | Dense $B^2$ | FWHT $B \log_2 B$ | Tỷ lệ |
|:---:|:---:|:---:|:---:|
| 512 | 262,144 | 4,608 | 57x |
| 1024 | 1,048,576 | 10,240 | 102x |
| 4096 | 16,777,216 | 49,152 | 341x |

Hiện thực CPU là `ggml_compute_forward_fwht_impl` (`ggml/src/ggml-cpu/ops.cpp`): kiểm tra $n$ là lũy thừa của 2, chia các hàng cho các thread, nhân trước với $1/\sqrt{n}$, rồi chạy các tầng butterfly (có nhánh SIMD, kể cả SVE).

So với phép nhân ma trận đi kèm, FWHT là rẻ về FLOP: với một tầng $d_{\text{out}} \times d_{\text{in}}$, phép nhân ma trận tốn $d_{\text{out}} d_{\text{in}}$ phép nhân-cộng mỗi token, còn FWHT tốn $d_{\text{in}} \log_2 B$. Nhưng ở batch 1, FWHT là một kernel riêng nằm trên đường găng (critical path) của **mọi** projection, nên chi phí khởi chạy và một lượt đọc ghi activation mới là vấn đề. Whitepaper Bonsai 2 27B ghi nhận phép xoay "vẫn là một trong những chi phí không phải nhân ma trận lớn nhất của một bước decode".

---

## Phần 4: Hiện thực trong llama.cpp

### 4.1. Biểu diễn phép xoay như một `mul_mat` có hint

Thay vì thêm một op mới vào ggml, llama.cpp biểu diễn phép xoay là một `ggml_mul_mat` với ma trận $H_B$ dày đặc, gắn hint `GGML_HINT_SRC0_IS_HADAMARD` (`src/llama-impl.h`):

```cpp
static inline ggml_tensor * llama_mul_mat_hadamard(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const auto n = rot->ne[0];                       // B
    ggml_tensor * res = ggml_is_contiguous(cur) ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n)
                                                : ggml_cont_2d   (ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat(ctx, rot, res);               // mỗi đoạn B phần tử nhân với H_B
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}
```

Phép reshape về $[B, \text{nelements}/B]$ hiện thực đúng $\mathbf{H} = I \otimes H_B$: mỗi đoạn $B$ phần tử liên tiếp của mỗi token là một cột độc lập. Ưu điểm của thiết kế:

- **Đúng trên mọi backend ngay cả khi không có kernel riêng**: backend nào chưa nhận diện hint vẫn thực hiện phép nhân dày đặc, cho kết quả đúng (chỉ chậm).
- **Backend có kernel FWHT bỏ qua `src0`**: CPU (`ggml_compute_forward_mul_mat` chuyển sang `ggml_compute_forward_fwht` khi thấy hint và không ở chế độ tham chiếu), CUDA (`fwht.cu`), Metal, Vulkan, SYCL (comment trong `fwht.hpp`: "src0 is not read at all"), BLAS và Hexagon.

Cái giá của thiết kế là tensor `prism.hadamard.<B>` kiểu F32 kích thước $B \times B$ phải được cấp phát: $4B^2$ byte, tức 1 MB với $B = 512$, 4 MB với $B = 1024$, và 64 MB với $B = 4096$ (bản 9B), cho mỗi buffer type có trọng số fold. Loader đặt tensor này vào buffer type của trọng số tương ứng nhưng **không bao giờ** vào buffer "extra" của CPU như `CPU_REPACK` (chỉ chấp nhận tensor nó repack được), đúng như comment trong `load_tensors_hadamard`.

### 4.2. Điểm chèn trong graph

Mọi phép nhân ma trận của mô hình đi qua `llm_graph_context::build_lora_mm` (dense) hoặc `build_lora_mm_id` (MoE). Cả hai gọi `build_hadamard_input(w, cur)` trước:

```cpp
ggml_tensor * llm_graph_context::build_hadamard_input(ggml_tensor * w, ggml_tensor * cur) const {
    if (!hdmd) return cur;
    const auto it = hdmd->rot.find(w);               // w có được fold không?
    if (it == hdmd->rot.end()) return cur;
    const auto & t = it->second;
    if (ggml_tensor * x = res->get_hdmd_input(cur, t.rot)) return x;   // dùng lại nếu đã tính
    ggml_tensor * x = cur;
    if (t.perm_rep > 1) { /* hoán vị head GDN, Phần 5 */ }
    if (t.signs) x = ggml_mul(ctx0, x, t.signs);      // D x
    x = llama_mul_mat_hadamard(ctx0, x, t.rot);       // H (D x)
    res->set_hdmd_input(cur, t.rot, x);
    return x;
}
```

Ba điểm thiết kế đáng học:

1. **Memoization theo cặp (input, rotation)**. ggml không có loại bỏ biểu thức con chung (CSE). Trong qwen35, `ffn_gate` và `ffn_up` cùng đọc một activation; `attn_q`, `attn_k`, `attn_v` cũng vậy. Commit #137 của fork đo được 144 trên 401 trọng số fold đang tính lại một biến đổi đã có, tức 36% số lần gọi FWHT mỗi token là thừa; loại bỏ chúng cho +3.1% decode trên Metal. Cơ chế này hiện đã có trong mainline (`llm_graph_result::get_hdmd_input`, `set_hdmd_input`).
2. **LoRA đọc activation gốc**. Trong `build_lora_mm`, phép nhân với trọng số fold dùng `cur_mm` (đã biến đổi), còn nhánh LoRA dùng `cur`: "a LoRA delta lives in the primal basis and reads `cur`". Nghĩa là adapter LoRA huấn luyện trên mô hình chưa xoay vẫn áp dụng đúng.
3. **Chỉ những đường đi qua `build_lora_mm` mới được phép fold**. Loader từ chối mọi tên tensor không thuộc danh sách `attn_q, attn_k, attn_v, attn_qkv, attn_gate, attn_output, ffn_gate, ffn_up, ffn_down, *_exps, *_shexp, ssm_out` và `output.weight`, và từ chối mọi kiến trúc ngoài `LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT`. Một tensor fold mà graph quên biến đổi sẽ cho ra sai số âm thầm, nên loại bỏ tận gốc khả năng đó là thiết kế phòng thủ đúng.

### 4.3. Bảng embedding và output gắn kết (tied)

`token_embd.weight` không được dùng trong phép nhân ma trận mà trong phép tra hàng (`ggml_get_rows`). Nếu bảng lưu các hàng ở cơ sở xoay (latent), runtime phải áp dụng **nghịch đảo** sau khi tra: tài liệu `docs/development/hadamard-tied-output.md` của fork viết "lookup reconstructs `s * (H z)`", và "the output projection uses the same stored rows after transforming its input as `H (s * h)`". Vì vậy:

- `prism.hadamard.inverse_weight_names` chỉ chấp nhận đúng một tên: `token_embd.weight`.
- Với mô hình tied embedding (2B, 4B của Bonsai 2 Small), cùng một ma trận latent vừa là bảng tra (cần nghịch đảo sau lookup) vừa là LM head (cần biến đổi trước matmul). Hợp đồng **version 2** (`prism.hadamard.version = 2`, `tied_output = true`, không có `output.weight`) diễn đạt điều đó; version 1 dành cho mô hình có LM head riêng (9B). Loader kiểm tra chặt: version 2 bắt buộc `tied_output = true` và ngược lại.

---

## Phần 5: Hoán vị value head của Gated DeltaNet

Đây là chi tiết tinh tế nhất. Với Qwen3.5 4B và 9B, tầng GDN có $n_v = 2 n_k$ value head. Đầu ra của khối GDN đi vào `ssm_out` (projection ra của linear attention) với $d_{\text{in}} = n_v \cdot h_d$ đặc trưng. Phép fold Hadamard được tính offline theo **thứ tự đặc trưng của Hugging Face**: nhóm theo key head, `[hd, rep, nk]` (viết theo thứ tự ggml, chiều nhanh nhất trước). Nhưng graph llama.cpp tạo ra các value head theo thứ tự **xếp lớp (tiled)** `[hd, nk, rep]`.

Vì phép xoay theo block trộn các đặc trưng trong cùng một block $B$, thứ tự đặc trưng không thể tùy ý: nếu áp dụng $D$ và $\mathbf{H}$ lên vector theo thứ tự sai, kết quả là một phép biến đổi khác hoàn toàn. Runtime giải quyết bằng một phép hoán vị trước khi nhân dấu (`llama_hadamard_transform::set_gdn_v_perm` trong `src/llama-model.h`):

```cpp
// tiled [hd, nk, rep] -> grouped [hd, rep, nk]
x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
x = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
```

với $h_d = d_{\text{in}} / n_v$, `perm_nk` $= n_k$, `perm_rep` $= n_v / n_k$. Metadata `prism.hadamard.gdn_v_grouped` báo cho runtime biết file được fold theo thứ tự grouped. Cũng chính vấn đề thứ tự này xuất hiện theo chiều ngược lại khi chuyển GGUF về checkpoint Hugging Face: các tensor nhánh value (`attn_qkv` phần V, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt`) phải được hoán vị từ thứ tự GGUF về thứ tự HF trước khi unfold.

---

## Phần 6: Hợp đồng metadata và các phép kiểm tra của loader

| Key GGUF | Giá trị hợp lệ | Kiểm tra trong `load_hparams_hadamard` |
|:---|:---|:---|
| `prism.hadamard.version` | 1 hoặc 2 | Khác thì từ chối; version 2 khi và chỉ khi `tied_output` |
| `prism.hadamard.block_size` | Lũy thừa của 2 | Phải chia hết chiều đầu vào của mọi trọng số fold |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` | Chuỗi khác thì từ chối |
| `prism.hadamard.axis` | `input-last-dimension` | Chuỗi khác thì từ chối |
| `prism.hadamard.sign_mode` | `identity` hoặc `explicit` | Với `explicit`, cần `sign_widths` khác rỗng |
| `prism.hadamard.sign_widths`, `sign_values` | Mỗi width là bội của block size; giá trị $\pm 1$ | Không trùng width, tổng độ dài khớp |
| `prism.hadamard.weight_names` | Danh sách tensor fold | Mỗi tên phải nằm trên đường matmul đã kiểm chứng |
| `prism.hadamard.inverse_weight_names` | Chỉ `token_embd.weight` | Tên khác thì từ chối |
| `prism.hadamard.gdn_v_grouped` | bool | Bật hoán vị head ở Phần 5 |

Mỗi độ rộng đầu vào khác nhau có một vector dấu riêng (`sign_widths`): ví dụ với bản 2B, hidden 2048 và intermediate 6144 ($= 12 \times 512$) là hai độ rộng, cần hai vector dấu, và cả hai đều chia hết cho block 512. Với bản 9B, block 4096 chia hết cả hidden 4096 lẫn intermediate 12288 ($= 3 \times 4096$).

**Vì sao runtime cũ sinh rác mà không báo lỗi.** Một runtime không biết các key `prism.hadamard.*` đơn giản bỏ qua chúng (GGUF cho phép key lạ). Nó nạp $W_f$ và tính $W_f x$ thay cho $W_f \mathbf{H} D x$. Vì $\mathbf{H}$ trộn đều mọi thành phần, $W_f x$ gần như không tương quan với $W x$, nên đầu ra là rác hoàn toàn chứ không phải "kém một chút". Đây là lý do model card yêu cầu bản mainline đủ mới khi chạy file `Q2_0` của Bonsai 2.

---

## Phần 7: Các tối ưu riêng của fork PrismML

Mainline đã có runtime đúng; fork thêm các phép gộp (fusion) để giảm chi phí ở batch 1:

| Tối ưu | Backend | Ý tưởng |
|:---|:---|:---|
| Gộp dấu vào lượt nạp của FWHT | Metal | Bỏ một lượt đọc ghi activation riêng cho `ggml_mul(x, signs)` |
| Gộp SwiGLU vào FWHT có dấu | Metal | Đầu vào của `ffn_down` là $\operatorname{SwiGLU}(\text{gate}, \text{up})$; tính nó ngay lúc nạp vào kernel FWHT, đầu ra GLU không đi qua bộ nhớ |
| Gộp FWHT với lượng tử activation Q8_1 | CUDA | Kernel `fwht_quantize_q8_1` (`quantize.cu`) xoay rồi lượng tử trong cùng một block thread, bỏ qua bước ghi activation F32 |
| FWHT block-per-row từ độ rộng 512 | CUDA | Song song hóa butterfly trên cả thread block thay vì một warp (+17% và +11% decode theo commit #166) |
| FWHT cho warp 64 | HIP | Sửa cho GPU AMD có wavefront 64 |

Từ góc nhìn roofline, các fusion này không giảm FLOP mà giảm **số lần đi qua bộ nhớ** và **số kernel** mỗi token, đúng hai yếu tố chi phối decode khi trọng số đã nhỏ tới 1.75 bpw.

---

## Phần 8: Bài tập

1. Chứng minh rằng nếu dùng $W_f = W \mathbf{H} D$ (đổi thứ tự $D$ và $\mathbf{H}$) thì activation phải được biến đổi thành $D \mathbf{H} x$. Viết lại `build_hadamard_input` cho quy ước đó.
2. Tính dung lượng tensor xoay F32 cho bản 9B ($B = 4096$) khi trọng số nằm trên hai GPU khác nhau. Đề xuất một cách biểu diễn khác tránh cấp phát $4B^2$ byte, và giải thích vì sao thiết kế hiện tại vẫn chấp nhận được.
3. Giả sử một backend không nhận diện hint và thực hiện phép nhân dày đặc. Ước lượng số FLOP tăng thêm mỗi token cho Bonsai 2 9B (32 tầng, hidden 4096, intermediate 12288, $B = 4096$), so với FWHT.
4. Vì sao bảng embedding không thể xử lý bằng cách "biến đổi activation" như các tầng tuyến tính? Liên hệ với sự khác nhau giữa `ggml_get_rows` và `ggml_mul_mat`.

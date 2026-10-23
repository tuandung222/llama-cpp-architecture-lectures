---
sidebar_position: 2
sidebar_label: "Bài 1: Toán học Quantization Error & Information Loss"
---

# Bài 1: Toán học Quantization Error & Information Loss

Bài này phân tích cơ sở toán học của quantization error: tại sao nén trọng số gây mất mát chất lượng, làm thế nào để đo lường, dự đoán, và tối thiểu hóa mất mát đó, và mỗi kết quả lý thuyết tương ứng với đoạn mã nào trong `ggml/src/ggml-quants.c`, `tools/imatrix/imatrix.cpp` và `src/llama-quant.cpp`.

---

## 1. Uniform Quantization Error

Với uniform quantization step $\Delta$ và giả thiết "high-resolution" (tín hiệu biến thiên nhiều trong mỗi khoảng lượng tử, không bị clipping), quantization error $\epsilon = x - \hat{x}$ xấp xỉ phân phối đều:

$$\epsilon \sim \text{Uniform}\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)$$

Mean Squared Error (MSE):

$$MSE = \mathbb{E}[\epsilon^2] = \frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2} \epsilon^2 \, d\epsilon = \frac{\Delta^2}{12}$$

Đọc công thức này: MSE tỷ lệ với bình phương của step size. Với $b$ bit phủ một dải $[-R, R]$, $\Delta = 2R / 2^b$, nên thêm 1 bit làm $\Delta$ giảm một nửa và MSE giảm 4 lần.

Trong block quantization của GGML, dải $R$ **không** cố định cho cả tensor mà được chọn lại cho mỗi block (32 hoặc 256 phần tử, hoặc sub-block 16/32 trong K-quants). Đây là đòn bẩy quan trọng nhất: với trọng số gần Gaussian $\mathcal{N}(0, \sigma^2)$, giá trị lớn nhất của một block 32 phần tử thường chỉ khoảng $2\sigma$ đến $2.5\sigma$, trong khi giá trị lớn nhất của cả tensor hàng triệu phần tử có thể là $5\sigma$ trở lên (chưa kể outlier). Scale cục bộ làm $\Delta$ nhỏ đi đáng kể so với scale toàn cục.

---

## 2. Signal-to-Quantization-Noise Ratio (SQNR)

$$SQNR = 10 \log_{10}\left(\frac{\sigma_x^2}{\sigma_\epsilon^2}\right) \approx 6.02\,b + 1.76 \text{ dB}$$

Trong đó $b$ là **số bit của mã lượng tử** ($2^b$ mức) và $\sigma_x^2$ là variance của tín hiệu gốc. Hằng số $1.76$ dB đúng cho tín hiệu hình sin phủ toàn dải; với phân phối khác, hằng số thay đổi (Gaussian với clipping tối ưu cho hằng số thấp hơn), nhưng độ dốc $6.02$ dB/bit giữ nguyên trong chế độ high-resolution vì $20\log_{10} 2 \approx 6.02$.

**Hệ quả**: Mỗi bit thêm vào tăng SQNR khoảng 6 dB, tương đương giảm MSE khoảng 4 lần.

Chú ý lấy $b$ là số bit của **mã**, không phải bpw hiệu dụng. Q4_K có bpw 4.5 nhưng mã của nó là 4 bit; 0.5 bpw còn lại là metadata (scale, min) không tạo thêm mức lượng tử. Bảng dưới đây dùng $b$ đúng và tỷ lệ MSE so với Q8_0, $4^{8 - b}$:

| Quant | Bit của mã ($b$) | SQNR theo công thức | MSE so với Q8_0 |
|:---|:---|:---|:---|
| Q8_0 | 8 | 49.9 dB | 1x (tham chiếu) |
| Q6_K | 6 | 37.9 dB | ~16x |
| Q5_K | 5 | 31.9 dB | ~64x |
| Q4_K | 4 | 25.8 dB | ~256x |
| Q3_K | 3 | 19.8 dB | ~1024x |
| Q2_K | 2 | 13.8 dB | ~4096x |

Bảng này cho thấy **xu hướng**, không phải giá trị đo thật. Lợi ích của scale cục bộ, của min (mô hình affine) và của tìm kiếm scale có trọng số (Phần 6) đều nằm ngoài công thức, nên SQNR thực tế của K-quants cao hơn con số "ngây thơ" ở trên.

---

## 3. Lloyd-Max Algorithm và lượng tử hóa phi tuyến

Lloyd-Max tìm optimal quantization levels cho một phân phối xác suất đã biết:

**Input**: PDF $f(x)$ của trọng số, số mức $L = 2^b$.

**Iterate**:
1. Cho trước các mức $y_1 < \dots < y_L$, boundaries tối ưu là trung điểm: $t_i = \frac{y_i + y_{i+1}}{2}$ (điều kiện nearest-neighbor).
2. Cho trước boundaries, mức tối ưu là trọng tâm của mỗi vùng (điều kiện centroid):

$$y_i = \frac{\int_{t_{i-1}}^{t_i} x f(x)\, dx}{\int_{t_{i-1}}^{t_i} f(x)\, dx}$$

Lặp đến khi hội tụ (MSE giảm đơn điệu sau mỗi bước, nên thuật toán hội tụ về một cực tiểu địa phương). Kết luận: khi phân phối không đều (thường là dạng chuông), các mức tối ưu **dày ở gần 0 và thưa ở hai đuôi**.

Đây chính là ý tưởng đằng sau `kvalues_iq4nl` trong `ggml/src/ggml-common.h`, bảng 16 mức cố định của IQ4_NL và IQ4_XS:

$$\lbrace -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 \rbrace$$

Khoảng cách giữa các mức liền kề tăng dần từ khoảng 11-12 gần 0 lên khoảng 23-24 ở hai đầu, đúng hình dạng mà Lloyd-Max dự đoán cho phân phối dạng chuông. Bảng `kvalues_fp4` của MXFP4/NVFP4 (các giá trị E2M1 nhân đôi: 0, 1, 2, 3, 4, 6, 8, 12) cũng là một lưới phi tuyến, nhưng do định dạng số thực quy định chứ không phải do tối ưu.

---

## 4. Block-wise Quantization Error Propagation

Xét mạng $h_l = \phi(W_l h_{l-1})$ với $\phi$ là hàm Lipschitz hằng số 1 (bỏ qua residual và norm để đơn giản). Thay $W_l$ bằng $W_l + \Delta W_l$ và khai triển bậc nhất, sai số ở đầu ra bị chặn bởi:

$$\|\Delta h_L\|_2 \lesssim \sum_{l=1}^{L} \left(\prod_{j=l+1}^{L} \|W_j\|_2\right) \|\Delta W_l\|_2 \, \|h_{l-1}\|_2$$

Trong đó $\Delta W_l$ là quantization error của weight matrix layer $l$. Đọc công thức này theo nghĩa thực tế: sai số ở layer $l$ được khuếch đại bởi chuẩn của **tất cả các layer phía sau**. Nếu chỉ nhìn công thức này, layer càng gần đầu vào càng nguy hiểm.

Trong transformer thực tế, residual connection và RMSNorm làm bức tranh phức tạp hơn: layer cuối ảnh hưởng trực tiếp đến logits, còn layer đầu thường có activation outlier. Thực nghiệm của tác giả K-quants dẫn đến một heuristic đối xứng trong `src/llama-quant.cpp`:

```c
auto use_more_bits = [](int i_layer, int n_layers) -> bool {
    return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
};
```

Preset Q4_K_M dùng hàm này để nâng `attn_v.weight` và `ffn_down.weight` lên Q6_K ở **1/8 số layer đầu, 1/8 số layer cuối** và một phần ba số layer ở giữa; các tensor attention khác vẫn là Q4_K. Không có preset K-quant nào giữ attention ở FP16.

---

## 5. Optimal Bit Allocation (reverse water-filling)

Giả sử có $N$ nhóm trọng số, nhóm $i$ có variance $\sigma_i^2$, và tổng ngân sách là $B = N\bar{b}$ bit. Trong chế độ high-resolution, MSE của nhóm $i$ là $c\,\sigma_i^2 2^{-2b_i}$. Bài toán

$$\min_{b_1, \dots, b_N} \sum_i c\,\sigma_i^2\, 2^{-2 b_i} \quad \text{s.t.} \quad \sum_i b_i = B$$

giải bằng Lagrange cho nghiệm

$$b_i = \bar{b} + \frac{1}{2} \log_2\left(\frac{\sigma_i^2}{\left(\prod_j \sigma_j^2\right)^{1/N}}\right)$$

Bản chất của công thức nằm ở việc: nhóm có variance gấp 4 lần trung bình hình học được thêm đúng 1 bit, và ở nghiệm tối ưu mọi nhóm có **cùng MSE**.

llama.cpp hiện thực ý tưởng này ở mức **thô**: không có phân bổ bit liên tục theo nhóm; thay vào đó `llama_tensor_get_type_impl()` chọn một kiểu rời rạc cho cả tensor (Q4_K, Q5_K, Q6_K...) theo loại tensor và vị trí layer. Trong một tensor, mọi trọng số có cùng số bit.

---

## 6. Weighted MSE và Importance Matrix

Phần 5 tối ưu MSE của **trọng số**, nhưng điều ta thực sự quan tâm là sai số của **đầu ra layer**. Với $y = W x$ và hàng $w$:

$$\mathbb{E}\left[\left((w - \hat{w})^{\top} x\right)^2\right] = (w - \hat{w})^{\top} C\, (w - \hat{w}), \qquad C = \mathbb{E}[x x^{\top}]$$

Giữ đường chéo của $C$:

$$\mathcal{L}(\hat{w}) \approx \sum_j \underbrace{\mathbb{E}[x_j^2]}_{\text{imatrix}_j} (w_j - \hat{w}_j)^2$$

Đây là định nghĩa chính xác của importance matrix trong llama.cpp: `tools/imatrix/imatrix.cpp` cộng dồn `acc[j] += x[j] * x[j]` cho mỗi cột $j$ của mỗi tensor trọng số trên dữ liệu hiệu chỉnh. Khi quantize, vector này được truyền xuống dưới tên `quant_weights`, và các hàm như `quantize_row_q4_K_impl()` dùng trọng số

$$w_l = \text{imatrix}_l \cdot \sqrt{\sigma^2 + x_l^2}, \qquad \sigma^2 = \frac{2}{256}\sum_{l} x_l^2$$

để chọn scale/min của từng sub-block (qua `make_qkx3_quants()`). Khi không có imatrix, trọng số dự phòng là $w_l = \sqrt{\sigma^2} + \lvert x_l \rvert$: ưu tiên phần tử có trị tuyệt đối lớn.

Hai điểm cần hiểu đúng:

- imatrix **không** phân bổ thêm bit cho trọng số quan trọng; nó thay đổi **hàm mục tiêu** khi chọn scale và mã, để sai số dồn vào các cột ít quan trọng.
- imatrix và I-quants là hai khái niệm độc lập: imatrix dùng được cho mọi kiểu (Q4_K, Q5_K, Q4_0...), I-quants là định dạng codebook. Chỉ IQ1_S, IQ2_XXS, IQ2_XS **bắt buộc** imatrix (`ggml_quantize_requires_imatrix()` trong `ggml.c`), vì ở dưới 2.5 bpw, tối ưu MSE không trọng số cho kết quả kém.

GPTQ đi xa hơn: dùng toàn bộ $C$ (không chỉ đường chéo) và bù sai số của cột vừa lượng tử sang các cột chưa lượng tử. AWQ dùng cùng thống kê $\mathbb{E}[\lvert x_j \rvert]$ nhưng để chọn hệ số scale theo kênh trước khi lượng tử.

---

## 7. Một kết quả đóng: scale tối ưu cho lượng tử hóa dấu (Q1_0)

Với Q1_0 (`quantize_row_q1_0_ref`), mỗi trọng số được thay bằng $d \cdot s_i$ với $s_i = \text{sign}(x_i)$. Cực tiểu $\sum_i (x_i - d s_i)^2$ theo $d$:

$$\frac{\partial}{\partial d}\sum_i (x_i - d s_i)^2 = -2\sum_i s_i x_i + 2 d \sum_i s_i^2 = 0 \;\Rightarrow\; d^{\ast} = \frac{1}{n}\sum_i \lvert x_i \rvert$$

Vậy scale tối ưu theo MSE là **trung bình trị tuyệt đối** (absmean), không phải max. Với $x \sim \mathcal{N}(0, \sigma^2)$, $d^{\ast} = \sigma\sqrt{2/\pi}$ và MSE tương đối là $1 - 2/\pi \approx 0.36$, tức SQNR chỉ khoảng 4.4 dB: đó là lý do lượng tử hóa 1-bit chỉ khả thi với mô hình được huấn luyện để chịu được nó.

---

## 💡 Đúc kết

Toán học quantization cho thấy:
- SQNR giảm khoảng 6 dB cho mỗi bit **mã** mất đi; metadata scale không tạo thêm mức mà thu hẹp dải động cục bộ.
- Lloyd-Max giải thích vì sao các mức phi tuyến (`kvalues_iq4nl`) tốt hơn mức đều cho phân phối dạng chuông.
- Sai số lan truyền qua các layer; heuristic `use_more_bits` của llama.cpp ưu tiên layer đầu, layer cuối và hai tensor `attn_v`, `ffn_down`.
- Phân bổ bit tối ưu cho nghiệm "cùng MSE"; llama.cpp xấp xỉ nó ở mức tensor. Importance matrix là đường chéo của $\mathbb{E}[x x^{\top}]$, dùng để đặt trọng số cho hàm sai số chứ không phải để phân bổ bit.

# Lộ trình thực nghiệm: Build, Benchmark và Deploy llama.cpp

Chào mừng đến với phần thực nghiệm chuyên sâu của khóa học llama.cpp. Trong khi các bài học trước cung cấp nền tảng lý thuyết về GGML, GGUF, quantization và inference engine, phần này đưa bạn trực tiếp vào terminal để build, cấu hình, benchmark và deploy thư viện trong môi trường thực tế.

Mỗi bài thực nghiệm được thiết kế như một lab hoàn chỉnh: có mục tiêu rõ ràng, step-by-step commands, giải thích output, và phần phân tích kết quả.

---

## 🗺️ Bản đồ lộ trình thực nghiệm

### [Bài 1: Build llama.cpp từ Source Code](exp_1_build_from_source.md)
* **Trọng tâm**: CMake, compiler flags, BLAS backends, CUDA/Metal/Vulkan.
* **Output**: Binary chạy được với đầy đủ backend support.

### [Bài 2: Custom Quantization với Importance Matrix](exp_2_custom_quantization_imatrix.md)
* **Trọng tâm**: Tạo importance matrix từ calibration data, quantize với imatrix.
* **Output**: File GGUF custom quant chất lượng cao hơn default.

### [Bài 3: Benchmark Suite - Đo hiệu năng toàn diện](exp_3_benchmark_suite.md)
* **Trọng tâm**: llama-bench, token generation speed, prompt processing, memory usage.
* **Output**: Báo cáo benchmark so sánh nhiều cấu hình.

### [Bài 4: Server Deployment - Docker và Production Setup](exp_4_server_deployment.md)
* **Trọng tâm**: Docker image, llama-server, OpenAI-compatible API, monitoring.
* **Output**: Server production-ready với health check và logging.

### [Bài 5: Perplexity Evaluation - Đánh giá chất lượng Quant](exp_5_quality_evaluation.md)
* **Trọng tâm**: llama-perplexity, wikitext benchmark, so sánh PPL giữa các quant.
* **Output**: Bảng PPL comparison và khuyến nghị chọn quant.

---

## 🎯 Mục tiêu đầu ra

Sau khi hoàn thành 5 bài thực nghiệm, học viên có khả năng:
1. Build llama.cpp từ source trên bất kỳ platform nào (Linux, macOS, Windows) với GPU backend.
2. Tạo custom quantization chất lượng cao sử dụng importance matrix.
3. Thiết kế và chạy benchmark suite toàn diện, phân tích kết quả khoa học.
4. Deploy llama.cpp server production-ready với Docker và monitoring.
5. Đánh giá chất lượng quantization bằng perplexity metric và đưa ra khuyến nghị.

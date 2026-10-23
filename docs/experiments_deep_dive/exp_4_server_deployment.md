# Bài 4: Server Deployment - Docker và Production Setup

## Phần 1. Mục tiêu

Deploy llama.cpp server production-ready với Docker, OpenAI-compatible API, health check, logging, và monitoring. Phù hợp cho cả self-hosted và cloud deployment.

---

## Phần 2. llama-server cơ bản

### Chạy server trực tiếp

```bash
./build/bin/llama-server \
    -m model-q4_k_m.gguf \
    -c 4096 \
    -np 4 \
    -t 8 \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    --metrics
```

`-c` là tổng dung lượng KV cache của server. Khi đặt `-np` tường minh (và không bật `-kvu`), nó được chia đều cho các slot (ở đây mỗi slot 1024 token); nếu để `-np` mặc định (auto), server dùng 4 slot với KV cache hợp nhất (unified), các slot dùng chung một pool `-c` token. `--metrics` bật endpoint Prometheus (mặc định tắt). Mặc định server chỉ lắng nghe `127.0.0.1:8080`, nên `--host 0.0.0.0` chỉ dùng khi đã có lớp bảo vệ phía trước (Phần 5). Server cũng phục vụ Web UI tích hợp tại `/`.

### Kiểm tra server

```bash
# Health check (endpoint công khai, không cần API key; /v1/health cũng dùng được)
curl http://localhost:8080/health
# Response: {"status":"ok"}  (HTTP 503 khi model đang nạp)

# Chat completion (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain GGUF format."}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }'
```

---

## Phần 3. Docker Deployment

### Cách 1: dùng image chính thức

Upstream phát hành sẵn image server cho nhiều backend: `ghcr.io/ggml-org/llama.cpp:server` (CPU), `:server-cuda` (CUDA 12), `:server-cuda13`, `:server-rocm`, `:server-vulkan`, `:server-intel` (SYCL)... Image server đặt sẵn `LLAMA_ARG_HOST=0.0.0.0`, `LLAMA_ARG_PORT=8080`, có `HEALTHCHECK` gọi `/health`, và entrypoint là `/app/llama-server`, nên chỉ cần truyền tham số:

```bash
# GPU (cần nvidia-container-toolkit)
docker run -d \
    --name llama-server \
    --gpus all \
    -p 8080:8080 \
    -v /path/to/models:/models \
    --restart unless-stopped \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/model.gguf -c 4096 -ngl 99 --metrics

# CPU only
docker run -d \
    --name llama-server-cpu \
    -p 8080:8080 \
    -v /path/to/models:/models \
    --cpus 8 \
    --memory 16g \
    --restart unless-stopped \
    ghcr.io/ggml-org/llama.cpp:server \
    -m /models/model.gguf -c 4096 -t 8 --metrics
```

### Cách 2: tự build image

Cách đáng tin cậy nhất là dùng chính Dockerfile của upstream trong `.devops/`, vốn đã xử lý build đa giai đoạn (image `devel` để biên dịch, image `runtime` để chạy), copy các thư viện `.so` và gói Web UI:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

# Target "server" của cuda.Dockerfile; CUDA_DOCKER_ARCH giới hạn kiến trúc để build nhanh hơn
docker build -t local/llama.cpp:server-cuda \
    --target server \
    --build-arg CUDA_DOCKER_ARCH=86 \
    -f .devops/cuda.Dockerfile .

# Bản CPU
docker build -t local/llama.cpp:server --target server -f .devops/cpu.Dockerfile .
```

Nếu tự viết Dockerfile, cần nhớ ba điểm mà `.devops/cuda.Dockerfile` đã làm: (1) biên dịch trong image `nvidia/cuda:*-devel-*` vì image `runtime` không có `nvcc`; (2) `BUILD_SHARED_LIBS` mặc định là ON, nên phải copy cả `libllama.so`, `libggml*.so` sang image chạy; (3) image chạy cần `libgomp1` (OpenMP) và `curl` cho `HEALTHCHECK`.

---

## Phần 4. Docker Compose cho Multi-Model

```yaml
services:
  llama-7b:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8081:8080"
    volumes:
      - ./models/7b:/models
    environment:
      - LLAMA_ARG_MODEL=/models/model-7b.gguf
      - LLAMA_ARG_CTX_SIZE=8192
      - LLAMA_ARG_N_GPU_LAYERS=99
      - LLAMA_ARG_ENDPOINT_METRICS=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  llama-70b:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8082:8080"
    volumes:
      - ./models/70b:/models
    environment:
      - LLAMA_ARG_MODEL=/models/model-70b.gguf
      - LLAMA_ARG_CTX_SIZE=8192
      - LLAMA_ARG_N_GPU_LAYERS=99
      - LLAMA_ARG_ENDPOINT_METRICS=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1', '2']
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - llama-7b
      - llama-70b
```

Mọi tham số dòng lệnh quan trọng đều có biến môi trường tương ứng dạng `LLAMA_ARG_*` (ghi trong cột "env" của `tools/server/README.md`); tham số dòng lệnh được ưu tiên hơn biến môi trường. Khóa `version:` ở đầu file Compose đã lỗi thời và có thể bỏ. Với model 70B chia trên hai GPU, `llama-server` tự chia layer theo `--split-mode layer` (mặc định); có thể chỉnh tỷ lệ bằng `-ts`.

Nếu chỉ cần một endpoint phục vụ nhiều model, `llama-server` còn có **router mode** (`--models-dir`, `--models-preset`, `--models-max`): một tiến trình nạp/gỡ model theo trường `model` của request.

---

## Phần 5. Authentication và Security

### API Key đơn giản

```bash
./build/bin/llama-server \
    -m model.gguf \
    --api-key "sk-your-secret-key-here" \
    --host 0.0.0.0 \
    --port 8080

# Hoặc đọc từ biến môi trường, tránh lộ key trong danh sách tiến trình
LLAMA_API_KEY="sk-your-secret-key-here" ./build/bin/llama-server -m model.gguf
```

Có thể truyền nhiều key (phân cách bằng dấu phẩy) hoặc đọc từ file bằng `--api-key-file`. `/health` và `/v1/health` luôn công khai để load balancer kiểm tra được.

Client request cần header:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer sk-your-secret-key-here" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Nginx Reverse Proxy

```nginx
# Trong khối http { ... }
limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;

upstream llama_backend {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location /v1/ {
        proxy_pass http://llama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;

        # Streaming (SSE): tắt buffering để token đến client ngay
        proxy_buffering off;

        # Rate limiting
        limit_req zone=api burst=10 nodelay;
    }
}
```

---

## Phần 6. Monitoring

### Prometheus metrics

Khi chạy với `--metrics` (hoặc `LLAMA_ARG_ENDPOINT_METRICS=1`), llama-server expose metrics tại `/metrics` (Prometheus text format); không có cờ này, endpoint bị tắt:

```bash
curl http://localhost:8080/metrics
```

Output mẫu (tên metric lấy từ mã nguồn `tools/server/server-task.cpp`):

```
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed, excluding cached tokens
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 15234
# HELP llamacpp:prompt_tokens_cached_total Number of prompt tokens reused from the cache
# TYPE llamacpp:prompt_tokens_cached_total counter
llamacpp:prompt_tokens_cached_total 48210
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 284571
# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 452.3
# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 312.4
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 3
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
```

> **Lưu ý**: Tên metric thực tế có prefix `llamacpp:` (dấu hai chấm, không phải dấu gạch dưới). Prometheus cho phép dấu hai chấm trong tên metric (thường dành cho recording rule), nên các truy vấn PromQL vẫn dùng nguyên tên này. Ngoài các metric trên còn có `prompt_seconds_total`, `tokens_predicted_seconds_total`, `n_decode_total`, `n_tokens_max`, `n_busy_slots_per_decode` và nhóm `spec_decode_*` cho speculative decoding.

### Grafana Dashboard

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llama-server'
    scrape_interval: 15s
    static_configs:
      - targets: ['llama-server:8080']
    # nếu server dùng --api-key, thêm:
    # authorization:
    #   credentials: sk-your-secret-key-here
```

Metrics quan trọng cần theo dõi:
- `llamacpp:prompt_tokens_total`: tổng prompt token count (counter).
- `llamacpp:tokens_predicted_total`: tổng generation token count (counter).
- `llamacpp:prompt_tokens_seconds`: prompt throughput (gauge).
- `llamacpp:predicted_tokens_seconds`: generation throughput (gauge).
- `llamacpp:requests_processing`: số request đang xử lý (gauge).
- `llamacpp:requests_deferred`: số request bị hoãn (gauge).

---

## Phần 7. Production Checklist

| Mục | Kiểm tra |
|:---|:---|
| Health check | `/health` trả về 200 |
| Graceful shutdown | SIGTERM xử lý đúng; kiểm thử hành vi với request đang stream trước khi đưa vào production |
| Memory limit | Docker `--memory` flag set đúng |
| GPU memory | VRAM đủ cho model + KV Cache của toàn bộ `-c` + compute buffer |
| Context per slot | Với `-np` tường minh, mỗi slot có `-c / -np` token: kiểm tra đủ dài cho request lớn nhất (hoặc dùng `-kvu`) |
| Logging | Stdout/stderr redirect đúng |
| TLS/SSL | HTTPS termination tại reverse proxy |
| Rate limiting | Nginx hoặc middleware |
| Restart policy | `--restart unless-stopped` hoặc Kubernetes |
| Monitoring | Prometheus scrape + Grafana alerts |
| Backup | Model files backup định kỳ |

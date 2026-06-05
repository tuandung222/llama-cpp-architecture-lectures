# Bài 4: Server Deployment - Docker và Production Setup

## 1. Mục tiêu

Deploy llama.cpp server production-ready với Docker, OpenAI-compatible API, health check, logging, và monitoring. Phù hợp cho cả self-hosted và cloud deployment.

---

## 2. llama-server cơ bản

### Chạy server trực tiếp

```bash
./llama-server \
    -m model-q4_k_m.gguf \
    -c 4096 \
    -t 8 \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99
```

### Kiểm tra server

```bash
# Health check
curl http://localhost:8080/health
# Response: {"status": "ok"}

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

## 3. Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

RUN apt update && apt install -y \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone và build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp . \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
    && cmake --build build --config Release -j$(nproc)

# Copy model vào image (hoặc mount volume)
COPY model.gguf /models/model.gguf

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./build/bin/llama-server", \
    "-m", "/models/model.gguf", \
    "-c", "4096", \
    "-t", "8", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "-ngl", "99"]
```

### Build và Run

```bash
# Build image
docker build -t llama-server .

# Run với GPU
docker run -d \
    --name llama-server \
    --gpus all \
    -p 8080:8080 \
    -v /path/to/models:/models \
    --restart unless-stopped \
    llama-server

# Run không GPU (CPU only)
docker run -d \
    --name llama-server-cpu \
    -p 8080:8080 \
    -v /path/to/models:/models \
    --cpus 8 \
    --memory 16g \
    --restart unless-stopped \
    llama-server-cpu
```

---

## 4. Docker Compose cho Multi-Model

```yaml
version: '3.8'

services:
  llama-7b:
    build: .
    ports:
      - "8081:8080"
    volumes:
      - ./models/7b:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/models/model-7b.gguf

  llama-70b:
    build: .
    ports:
      - "8082:8080"
    volumes:
      - ./models/70b:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1', '2']
              capabilities: [gpu]
    environment:
      - MODEL_PATH=/models/model-70b.gguf

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

---

## 5. Authentication và Security

### API Key đơn giản

```bash
./llama-server \
    -m model.gguf \
    --api-key "sk-your-secret-key-here" \
    --host 0.0.0.0 \
    --port 8080
```

Client request cần header:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Authorization: Bearer sk-your-secret-key-here" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### Nginx Reverse Proxy

```nginx
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
        
        # Rate limiting
        limit_req zone=api burst=10 nodelay;
    }
}
```

---

## 6. Monitoring

### Prometheus metrics

llama-server expose metrics tại `/metrics` endpoint (Prometheus format):

```bash
curl http://localhost:8080/metrics
```

Output mẫu:

```
llama_requests_total{method="POST",path="/v1/chat/completions"} 1523
llama_tokens_generated_total 284571
llama_prompt_processing_seconds_sum 45.23
llama_generation_seconds_sum 312.45
llama_kv_cache_usage_percent 0.67
```

### Grafana Dashboard

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llama-server'
    scrape_interval: 15s
    static_configs:
      - targets: ['llama-server:8080']
```

Metrics quan trọng cần theo dõi:
- `llama_requests_total`: tổng request count.
- `llama_kv_cache_usage_percent`: KV Cache utilization (alert > 90%).
- `llama_generation_seconds`: latency P50/P95/P99.
- `llama_tokens_generated_total`: throughput theo thời gian.

---

## 7. Production Checklist

| Mục | Kiểm tra |
|:---|:---|
| Health check | `/health` trả về 200 |
| Graceful shutdown | SIGTERM xử lý đúng, không drop request |
| Memory limit | Docker `--memory` flag set đúng |
| GPU memory | VRAM đủ cho model + KV Cache max |
| Logging | Stdout/stderr redirect đúng |
| TLS/SSL | HTTPS termination tại reverse proxy |
| Rate limiting | Nginx hoặc middleware |
| Restart policy | `--restart unless-stopped` hoặc Kubernetes |
| Monitoring | Prometheus scrape + Grafana alerts |
| Backup | Model files backup định kỳ |

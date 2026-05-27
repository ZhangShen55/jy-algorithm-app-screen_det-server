# 屏幕倾斜角度检测服务（Screen Tilt Detection）

基于 **FastAPI + OpenCV + YOLO** 的 HTTP 检测服务：

- **倾斜检测** `/detect_tilt`：OpenCV CPU 线段角度
- **屏幕类型检测** `/detect_screen`：YOLO GPU（`model/screen.pt`）

| 项目 | 说明 |
|------|------|
| 服务名 | `tilt-detection-service` |
| 默认端口 | `8880`（直连 Uvicorn，无 Nginx） |
| 倾斜推理 | **CPU**（OpenCV headless） |
| 屏幕推理 | **GPU**（Ultralytics YOLO，启动预加载 + warmup） |
| 配置 | 根目录 `config.toml`（Docker 建议挂载） |
| Agent 文档 | [AGENT.md](./AGENT.md) |
| 接口文档 | [docs/API接口文档.md](./docs/API接口文档.md) |

---

## 目录

- [API 接口文档](docs/API接口文档.md)
- [功能特性](#功能特性)
- [算法原理](#算法原理)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [本地开发（Conda）](#本地开发conda)
- [API 说明](#api-说明)
- [配置说明](#配置说明)
- [Docker 部署](#docker-部署)
- [存储与资源风险](#存储与资源风险)
- [压测与性能](#压测与性能)
- [测试记录](#测试记录)
- [常见问题](#常见问题)

---

## 功能特性

- 单图 / 批量检测：JSON `{"images": "<base64>"}` 或数组
- **异步接口** `/detect_tilt`、`/detect_screen`：CPU/GPU 推理在线程池执行
- YOLO **启动预加载 + GPU warmup**，health 返回 `ready` / `warmed_up`
- 健康检查、运行时配置查询、**热重载** `config.toml`（`[detection]` 等，GPU/worker 需重启）
- 请求访问日志、应用日志（**轮转**）、`X-Request-ID` 追踪
- Docker：`pytorch/cuda11.8` 基础镜像 + PyArmor 混淆 + `docker run` 部署

> 接口返回的 `start_time`、`end_time` 为 **北京时间** 对应的毫秒时间戳字符串。

---

## 算法原理

核心实现在 `app/services/tilt_detector.py`：

1. **解码**：Base64 → PIL RGB → OpenCV BGR；支持 `data:image/...;base64,` 前缀；限制最大体积（默认 10MB）
2. **预处理**：灰度 → 高斯模糊 → Canny 边缘
3. **线段检测**：`cv2.createLineSegmentDetector` 提取线段
4. **筛选**：保留长度 ≥ 图宽 × `min_line_length_ratio` 的线段；角度落在水平带（约 ±30°）或垂直带（约 60°–120°）
5. **聚合**：按角度排序后截取中间段（`trim_start_ratio`–`trim_end_ratio`），以线段长度为权重求平均，得到 **整体倾斜角** `angle`（绝对值，单位：度）
6. **判定**：`angle > tilt_threshold` 时内部记为倾斜（`is_tilted`）；API 响应主要返回 `angle`，阈值可通过配置调整

`model/classes.txt` 为相关业务类别标签（如 `askew-screen`、`normal-screen` 等），**当前倾斜检测 API 未加载该分类模型**，仅作业务参考。

---

## 项目结构

```
jy-algorithm-app-screen_det-server/
├── app/
│   ├── main.py              # FastAPI 入口、中间件、异常处理
│   ├── api/v1/
│   │   ├── router.py        # 聚合各功能路由
│   │   ├── tilt.py          # 倾斜检测
│   │   ├── screen.py        # 屏幕类型 YOLO 检测
│   │   ├── health.py        # 健康检查
│   │   ├── config.py        # 配置查询 / 热重载
│   │   └── common.py        # 公共工具（时间戳等）
│   ├── core/
│   │   ├── config.py        # 读取 config.toml
│   │   ├── logging.py       # 日志初始化
│   │   └── state.py         # 请求计数等运行时状态
│   ├── schemas/             # Pydantic 模型
│   └── services/
│       ├── tilt_detector.py
│       ├── screen_detector.py
│       └── yolo_compat.py
├── config.toml
├── requirements.txt         # 本地 Conda
├── requirements-docker.txt  # Docker 构建（不含 torch）
├── Dockerfile
├── start.sh
├── AGENT.md                 # AI Agent / 维护说明
├── model/screen.pt          # YOLO 权重（打入镜像）
├── scripts/                 # 验收与本地调试
├── test/
│   ├── tilt_img/            # detect_tilt 测试图
│   ├── ok_img/              # detect_screen 正常样例
│   └── error_img/           # detect_screen 异常样例
└── logs/                    # 运行日志（建议 Docker 挂载）
```

路由在 `app/main.py` 中挂载了两次（`prefix=/api/v1` 与无前缀），因此以下路径等价：

| 用途 | 路径示例 |
|------|----------|
| 健康检查 | `/health` 或 `/api/v1/health` |
| 倾斜检测 | `/detect_tilt` 或 `/api/v1/detect_tilt` |
| 屏幕检测 | `/detect_screen` 或 `/api/v1/detect_screen` |

---

## 环境要求

- **Python**：3.10+（本地 Conda `screen_det`）；Docker 为 Python 3.11（PyTorch 镜像）
- **GPU**：`/detect_screen` 需要 NVIDIA 驱动 + `--gpus`；`/detect_tilt` 仅 CPU
- **Docker**：NVIDIA Container Toolkit；基础镜像 `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime`

---

## 本地开发（Conda）

### 1. 激活环境并安装依赖

```bash
cd /root/workspace/jy-algorithm-app-screen_det-server

conda activate screen_det
pip install -r requirements.txt
```

### 2. 调整 GPU（可选）

编辑 `config.toml` 中 `[gpu].device_id`；单 GPU 建议 `[server].workers = 1`。

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8880 --workers 1
# 或使用 start.sh（与 Docker 相同逻辑）
bash start.sh
```

### 4. 健康检查

```bash
curl http://127.0.0.1:8880/
curl http://127.0.0.1:8880/health
curl http://127.0.0.1:8880/api/v1/health
```

### 5. 使用测试图片

测试数据在 `test/` 目录：

| 目录 | 接口 |
|------|------|
| `test/tilt_img/` | `/detect_tilt` |
| `test/ok_img/`、`test/error_img/` | `/detect_screen` |

验收脚本（自动拉起服务、跑用例、关闭）：

```bash
bash scripts/run_deploy_verify.sh
```

---

## API 说明

### 根路径 `GET /`

返回服务名、版本及各接口路径。

### 健康检查 `GET /health`

```json
{
  "status": "success",
  "ready": true,
  "elapsed_time": "0h 5m 12s",
  "total_requests": 42,
  "memory_mb": 1451.0,
  "gpu": {
    "enabled": true,
    "device_id": "1",
    "device_id_config": "1",
    "yolo_device_resolved": 1,
    "require_gpu": true,
    "tilt_inference_device": "cpu"
  },
  "screen_model": {
    "loaded": true,
    "warmed_up": true,
    "weights": "/app/model/screen.pt",
    "device": 1,
    "gpu_memory_mb": 512.0
  }
}
```

未就绪时 `ready: false`，HTTP 503。

### 倾斜检测 `POST /detect_tilt`

| 请求方式 | Content-Type | Body |
|----------|--------------|------|
| 纯 Base64 | `text/plain` | 图片 Base64 字符串（可无 data URL 前缀） |
| JSON | `application/json` | `{"images": "<base64>"}`；可选 `"tilt_threshold": 1.5` 覆盖配置 |

`tilt_threshold` 默认读 `config.toml` 的 `[detection].tilt_threshold`；`text/plain` 请求无法传该参数，使用默认值。

**成功响应**

```json
{
  "code": 200,
  "start_time": "1753791280207",
  "end_time": "1753791280225",
  "msg": "检测完成",
  "tilt_threshold": 1.5,
  "result": {
    "is_tilted": true,
    "angle": 2.35,
    "cost_ms": 37.59
  }
}
```

`is_tilted`：`angle > tilt_threshold` 时为 `true`。

### 屏幕类型检测 `POST /detect_screen`（YOLO）

使用 `model/screen.pt`，仅返回 **label 0–3**（蓝/黑/白/正常屏）。推理设备由 `config.toml` 的 `[gpu].device_id` 指定。

**请求**（`Content-Type: application/json`）

单图：

```json
{
  "images": "base64字符串",
  "conf": 0.25,
  "iou": 0.45
}
```

多图：`images` 改为字符串数组；`conf` / `iou` 可选，默认读 `[screen_detection]`。

**成功响应**

```json
{
  "code": 200,
  "start_time": "1753791280207",
  "end_time": "1753791280274",
  "msg": "检测完成",
  "conf": 0.25,
  "iou": 0.45,
  "total": 1,
  "results": [
    {
      "index": 0,
      "cost_ms": 48.6,
      "primary": {
        "label": 3,
        "confidence": 0.926,
        "box": [936, 55, 1697, 493]
      },
      "detections": [
        {
          "label": 3,
          "confidence": 0.926,
          "box": [936, 55, 1697, 493]
        }
      ]
    }
  ]
}
```

| label | 含义 |
|-------|------|
| 0 | blue-screen |
| 1 | black-screen |
| 2 | white-screen |
| 3 | normal-screen |

`box` 为 `[x1, y1, x2, y2]`：左上角 + 右下角，像素坐标。

### 配置 `GET /config`

返回当前 `app`、`server`、`gpu`、`detection`、`runtime` 配置快照。

### 重载配置 `POST /config/reload`

重新读取 `config.toml` 中的 `[detection]` 等配置，无需重启。

```bash
curl -X POST http://127.0.0.1:8880/config/reload
```

### 错误响应

- `400`：`{"code": 400, "msg": "..."}`（参数/Base64/图片无效等）
- `500`：`{"code": 500, "msg": "..."}`

---

## 配置说明

所有配置位于根目录 `config.toml`。

### 应用与服务

```toml
[app]
name = "tilt-detection-service"
version = "1.0.0"
debug = false
api_prefix = "/api/v1"

[server]
host = "0.0.0.0"
port = 8880
workers = 1
```

### GPU 与 YOLO 设备

```toml
[gpu]
enabled = true
device_id = "1"           # YOLO GPU；--gpus device=N 时容器内可能映射为 cuda:0
require_gpu = true        # 启动预加载时校验 CUDA

[screen_detection]
preload_at_startup = true # 随服务启动加载 + GPU warmup
weights_path = "model/screen.pt"
conf = 0.25
iou = 0.45
allowed_class_ids = [0, 1, 2, 3]
max_batch_size = 16
```

| 模块 | 设备 |
|------|------|
| 倾斜检测 `/detect_tilt` | **CPU**（OpenCV） |
| 屏幕检测 `/detect_screen` | **`[gpu].device_id` 指定 GPU**；`enabled=false` 时用 CPU |

### 检测参数

| 键 | 默认值 | 含义 |
|----|--------|------|
| `tilt_threshold` | `1.5` | 判定倾斜的角度阈值（度） |
| `min_line_length_ratio` | `0.1` | 最小线段长度 = 图宽 × 该比例 |
| `min_valid_lines` | `5` | 至少多少条有效线段才计算角度 |
| `trim_start_ratio` / `trim_end_ratio` | `0.2` / `0.8` | 角度样本截断区间 |
| `gaussian_kernel_size` | `5` | 高斯核（自动调整为奇数 ≥3） |
| `canny_threshold1` / `canny_threshold2` | `50` / `150` | Canny 双阈值 |
| `horizontal_angle_min/max` | `-30` / `30` | 水平参考线角度范围 |
| `vertical_angle_min/max` | `60` / `120` | 垂直参考线角度范围 |

```toml
[detection]
tilt_threshold = 1.5
# ... 其余见 config.toml
```

### 日志与运行时

```toml
[logging]
level = "INFO"
log_dir = "logs"
access_log = "access.log"
app_log = "app.log"

[runtime]
max_image_bytes = 10485760   # 10MB
```

---

## Docker 部署

### 构建

```bash
docker build -t jy-algorithm-app-screen_det-server:v1.0_260525 .
```

- 基础镜像：`pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime`（含 torch+cu118）
- 依赖：`requirements-docker.txt`（不含 torch，避免覆盖基础镜像）
- 业务代码：构建阶段 PyArmor 混淆
- 模型：`model/screen.pt` 打入镜像；`config.toml` **运行时挂载**

### 运行

```bash
docker run -d \
  --name tilt-api \
  --restart unless-stopped \
  --gpus all \
  -p 8880:8880 \
  -v /path/to/config.toml:/app/config.toml:ro \
  -v /path/to/logs:/app/logs \
  jy-algorithm-app-screen_det-server:v1.0_260525
```

### 验证

```bash
docker logs tilt-api | grep -E "preload|warmup"
curl -s http://127.0.0.1:8880/api/v1/health
nvidia-smi
```

| 变更类型 | 操作 |
|----------|------|
| 仅 `config.toml` | `docker restart tilt-api` |
| 代码 / Dockerfile / start.sh | 重新 `docker build` + `docker run` |

> PyPI 构建失败时，可临时本地 `pip download` 到 `wheels/` 并修改 Dockerfile 的 pip 安装方式（非默认路径）。

---

## 存储与资源风险

本项目 **无 Redis/磁盘缓存**，不存在典型缓存击穿；需关注以下资源边界：

| 类型 | 机制 | 上限 / 说明 |
|------|------|-------------|
| **日志磁盘** | `RotatingFileHandler` | 每文件 10MB × 11 份 × 2 个日志 ≈ 220MB |
| **单图内存** | `max_image_bytes` | 默认 10MB / 张 |
| **批量** | `max_batch_size` | 默认 16 张 / 请求 |
| **YOLO 缓存** | `YOLO_CONFIG_DIR` → `logs/.ultralytics` | 避免写 `/tmp` |
| **并发内存** | 无全局限流 | 高并发大图为主要风险，建议网关限流 |

单 GPU 生产建议：`workers = 1`，避免多 worker 重复占用显存。

---

## 压测与性能

压测前确认服务正常：

```bash
curl http://127.0.0.1:8880/health
```

**完整企业级套件**（预热、异步/同步阶梯、四图单测、批量混合、长稳）：

```bash
bash scripts/run_enterprise_benchmark.sh
# 指定远程地址
BASE_URL=http://10.80.5.197:8880 bash scripts/run_enterprise_benchmark.sh
```

结果保存在 `benchmark_reports/<时间戳>/`。关注输出中的 `failed`、`qps`、`latency_ms.p95`、`latency_ms.p99`。

压测期间可观察：

```bash
docker logs -f tilt-api    # Docker 部署时
watch -n 1 nvidia-smi
```

---

## 测试记录

在 **Conda 环境 `screen_det`**、项目根目录 **`text*.jpg`** 上于 **2026-05-20** 执行验证（默认 `tilt_threshold = 1.5`）。

### 算法直连（不经过 HTTP）

| 图片 | angle (°) | 是否超过阈值 | 说明 |
|------|-----------|--------------|------|
| text0.jpg | 2.35 | 是 | 明显倾斜样例 |
| text1.jpg | 0.52 | 否 | |
| text2.jpg | 0.06 | 否 | |
| text3.jpg | 0.84 | 否 | |

### HTTP 异步接口 `POST /detect_tilt`

| 图片 | angle | cost_ms（约） |
|------|-------|----------------|
| text0.jpg | 2.35 | 38 |
| text1.jpg | 0.52 | 139 |
| text2.jpg | 0.06 | 136 |
| text3.jpg | 0.84 | 137 |

JSON 请求 `{"images": "<base64>"}`：`result.angle` / `result.cost_ms` 与上表一致。

### 轻量压测（text0.jpg，20 请求，并发 5）

| 指标 | 值 |
|------|-----|
| success / failed | 20 / 0 |
| QPS | ~94.3 |
| 延迟 avg / p95 | ~52ms / ~55ms |

> 首次请求或冷启动可能略慢；生产评估建议先预热再跑 `run_enterprise_benchmark.sh`。

---

## 常见问题

**Q：`[gpu].require_gpu` 做什么？**  
A：启动 YOLO 预加载时校验 CUDA 与 `device_id` 合法性；失败则容器/worker 无法进入 ready。

**Q：Docker 里 `device_id=1` 不生效？**  
A：若使用 `--gpus '"device=1"'`，容器内仅可见 `cuda:0`，服务会自动映射；多卡 `--gpus all` 时按编号 0/1/2 使用。

**Q：nginx 目录去哪了？**  
A：当前架构为 Uvicorn 直连 `-p 8880:8880`，不再需要 Nginx 反向代理。

**Q：返回 `有效参考线段不足` 或 `未检测到有效直线`？**  
A：画面缺少清晰水平/垂直边缘，可调低 `canny_threshold*`、`min_line_length_ratio` 或 `min_valid_lines`（需结合误检率评估）。

**Q：如何修改倾斜判定阈值？**  
A：修改 `config.toml` 中 `[detection].tilt_threshold`，然后 `POST /config/reload`。

**Q：Postman 如何测单图？**  
A：`POST http://<host>:8880/detect_tilt`，Body 选 raw，类型 text，直接粘贴 Base64（不要加 JSON 引号）。

---

## 依赖版本

见 `requirements.txt` / `requirements-docker.txt`：

- fastapi、uvicorn、opencv-python-headless、ultralytics
- Docker 镜像内 torch 由 `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime` 提供

---

## 许可证与联系

企业内部算法服务项目；部署地址与网络策略以实际环境为准。历史文档中的路径 `detect_tilt` 与对外 IP 仅作示例，请以当前仓库与运维配置为准。

# AGENT.md — 屏幕倾斜 + 屏幕类型检测服务

供 Cursor / AI Agent 快速理解本仓库的上下文、约束与常见操作。

---

## 项目概述

| 项 | 值 |
|----|-----|
| 服务名 | `tilt-detection-service` |
| 框架 | FastAPI + Uvicorn |
| 默认端口 | `8880` |
| 配置 | 根目录 `config.toml`（Docker 运行时挂载） |
| 模型 | `model/screen.pt`（YOLO，10 类训练，API 仅返回 0–3） |

**两个核心接口：**

| 接口 | 实现 | 设备 |
|------|------|------|
| `POST /detect_tilt` | OpenCV 线段角度 | CPU |
| `POST /detect_screen` | Ultralytics YOLO | GPU（`[gpu].device_id`） |
| `POST /detect_inspect` | 倾斜 + 屏幕组合（单图 `image`） | CPU + GPU |

路由双挂载：`/health` 与 `/api/v1/health` 等价（见 `app/main.py`）。

---

## 目录结构（维护时关注）

```
app/
  main.py                 # 入口、startup YOLO 预加载、中间件
  api/v1/                 # tilt / screen / health / config
  services/
    tilt_detector.py      # 倾斜算法
    screen_detector.py    # YOLO 加载、warmup、推理
    yolo_compat.py        # 旧权重 AAttn 兼容
  core/config.py          # config.toml 读取（lru_cache）
config.toml               # 运行时配置（Docker 挂载）
model/screen.pt           # 打进 Docker 镜像
requirements.txt          # 本地 Conda 开发
requirements-docker.txt   # Docker 构建（不含 torch）
Dockerfile                # PyArmor 混淆 + pytorch/cuda11.8 基础镜像
start.sh                  # 读 config 启 uvicorn
scripts/                  # 验收、单图检测等（不进生产镜像）
test/                     # 测试图与验收报告
  tilt_img/               # detect_tilt
  ok_img/ error_img/      # detect_screen
```

**已移除、勿再引用：** `nginx/`、`docker-compose.yml`、默认离线 `wheels/` 构建。

---

## Docker 部署（当前标准方式）

```bash
docker build -t jy-algorithm-app-screen_det-server:v1.0_260525 .

docker run -d \
  --name tilt-api \
  --restart unless-stopped \
  --gpus all \
  -p 8880:8880 \
  -v /path/to/config.toml:/app/config.toml:ro \
  -v /path/to/logs:/app/logs \
  jy-algorithm-app-screen_det-server:v1.0_260525
```

要点：

- 基础镜像：`pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime`（torch+cu118 预装）
- 业务代码在构建阶段 **PyArmor 混淆**；混淆阶段 Python 版本须与运行镜像一致
- `config.toml` **不**打入镜像，必须挂载
- `model/screen.pt` 打入镜像
- 改 `config.toml` → `docker restart`；改代码/`start.sh`/`Dockerfile` → **rebuild**

### 启动预加载

`app/main.py` 的 `startup` 会：

1. 加载 YOLO 权重
2. GPU warmup（dummy predict），使显存在启动阶段即占用
3. 失败则 worker 退出（fail-fast）

health 字段：`ready`、`screen_model.warmed_up`、`screen_model.gpu_memory_mb`。

### device_id 与 Docker GPU

- `--gpus all`：容器内可见多卡，`device_id="1"` → `cuda:1`
- `--gpus '"device=1"'`：仅暴露宿主机 GPU1，容器内重编号为 **`cuda:0`**，代码会自动映射

---

## 本地开发

```bash
conda activate screen_det
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8880 --workers 1
```

验收脚本：

```bash
bash scripts/run_deploy_verify.sh
```

---

## 配置热重载范围

| 配置段 | `POST /config/reload` | 需重启 |
|--------|----------------------|--------|
| `[detection]` | ✅ | |
| `[screen_detection]` 部分字段 | ✅ | |
| `[gpu]`、`[server]`、`workers` | | ✅ |

---

## 存储与资源风险（Agent 修改代码时注意）

### 已有防护

| 风险点 | 现状 |
|--------|------|
| 日志撑满磁盘 | `RotatingFileHandler`：`max_bytes=10MB`，`backup_count=10`，单文件约 110MB 上限 |
| 单图过大 | `[runtime].max_image_bytes` 默认 10MB |
| 批量过大 | `[screen_detection].max_batch_size` 默认 16 |
| Ultralytics/matplotlib 写 /tmp | `start.sh` 将 `YOLO_CONFIG_DIR`、`MPLCONFIGDIR` 定向到 `logs/` 下 |

### 仍存在的风险（无 Redis/无全局限流）

| 风险 | 说明 | 建议 |
|------|------|------|
| **内存击穿** | 高并发 × 大 Base64 解码，峰值内存 ≈ 并发数 × batch × 10MB | 网关限流；生产调低 `max_batch_size` |
| **GPU 显存** | 单 worker 单份模型；误设 `workers>1` 会多份 YOLO | 单 GPU 保持 `workers=1` |
| **挂载 logs 卷** | 轮转有上限，但 `.ultralytics` 等子目录可能缓慢增长 | 定期清理或监控磁盘 |
| **无请求队列** | 突发流量直接进 threadpool | 前置网关 rate limit |

本项目 **无 Redis/无磁盘缓存**，不存在经典「缓存击穿/穿透」；主要关注 **内存与磁盘日志**。

---

## 代码修改约定

1. **最小 diff**：只改与任务相关的文件
2. **Docker 与本地依赖分离**：torch 只在基础镜像 / `requirements.txt`，不在 `requirements-docker.txt`
3. **混淆**：改 `app/` 后 Docker 必须 rebuild（obfuscator 阶段）
4. **不要恢复** nginx / docker-compose，除非用户明确要求
5. **端口统一 8880**
6. **中文注释**仅用于非显而易见的业务/部署逻辑

---

## 常见排查

```bash
docker logs tilt-api | grep -E "preload|warmup|failed"
curl -s http://127.0.0.1:8880/api/v1/health | python3 -m json.tool
nvidia-smi
```

| 现象 | 可能原因 |
|------|----------|
| 启动循环重启 | GPU 不可用、`device_id` 越界、模型路径错误 |
| `ready: false` | preload/warmup 未完成或失败 |
| 首包仍慢 | 未 rebuild 到带 warmup 的版本 |
| PyArmor `_PyFloat_Pack8` | 混淆与运行 Python 小版本不一致 |

---

## 测试数据

| 目录 | 用途 |
|------|------|
| `test/tilt_img/` | `/detect_tilt` |
| `test/ok_img/` | `/detect_screen` 正常样例 |
| `test/error_img/` | `/detect_screen` 异常样例 |

---

## 关键文件清单（部署）

- `Dockerfile`
- `start.sh`
- `requirements-docker.txt`
- `config.toml`（挂载）
- `model/screen.pt`（镜像内）

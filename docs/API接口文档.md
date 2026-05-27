# 屏幕倾斜 + 屏幕类型检测服务 — 接口文档

| 项目 | 说明 |
|------|------|
| 服务名 | tilt-detection-service |
| 版本 | v1.0.0 |
| 默认端口 | 8880 |
| 基础地址 | `http://<host>:8880` |
| 路由前缀 | `/` 与 `/api/v1/` **等价**（以下 URL 均给出两种写法） |
| 配置来源 | 根目录 `config.toml`（Docker 建议挂载） |

---

## 目录

1. [通用约定](#通用约定)
2. [接口列表](#接口列表)
3. [接口明细表（填表格式）](#接口明细表填表格式)
4. [错误码汇总](#错误码汇总)
5. [附录：label 枚举与配置项](#附录label-枚举与配置项)

---

## 通用约定

### 请求头

| Header | 说明 |
|--------|------|
| `Content-Type` | `detect_tilt`：支持 `application/json` 或 `text/plain`；`detect_screen`：**必须** `application/json` |
| `X-Request-ID` | 可选；未传时服务端自动生成，并在响应头回传 |

### 时间字段

`start_time`、`end_time` 为 **北京时间** 对应的毫秒时间戳字符串。

### 图片入参

- 编码：Base64 字符串
- 支持 `data:image/jpeg;base64,` 等 data URL 前缀（自动剥离）
- 单张解码后体积上限：默认 **10MB**（`config.toml` → `[runtime].max_image_bytes`）

### 成功 / 失败响应

| HTTP 状态码 | Body 结构 |
|-------------|-----------|
| 200 | 各接口业务 JSON（含 `code: 200`） |
| 400 | `{"code": 400, "msg": "..."}` |
| 500 | `{"code": 500, "msg": "..."}` |
| 503 | 仅 `/health`：YOLO 未就绪时 |

---

## 接口列表

| 序号 | 功能类型 | 方法 | URL（任选其一） | 说明 |
|------|---------|------|----------------|------|
| 1 | 服务信息 | GET | `/` 、`/api/v1/` | 返回服务名、版本、各接口路径 |
| 2 | 健康检查 | GET | `/health` 、`/api/v1/health` | 运行状态、GPU、YOLO 预加载状态 |
| 3 | 倾斜检测 | POST | `/detect_tilt` 、`/api/v1/detect_tilt` | OpenCV CPU 线段角度检测 |
| 4 | 屏幕检测 | POST | `/detect_screen` 、`/api/v1/detect_screen` | YOLO GPU 屏幕类型检测 |
| 5 | 配置查询 | GET | `/config` 、`/api/v1/config` | 返回当前配置快照 |
| 6 | 配置重载 | POST | `/config/reload` 、`/api/v1/config/reload` | 热重载部分配置（见说明） |

---

## 1. 服务信息

**URL：** `GET /` 或 `GET /api/v1/`

**请求：** 无

**响应示例：**

```json
{
  "service": "tilt-detection-service",
  "version": "1.0.0",
  "health": "/api/v1/health",
  "detect_tilt": "/api/v1/detect_tilt",
  "detect_screen": "/api/v1/detect_screen"
}
```

---

## 2. 健康检查

**URL：** `GET /health` 或 `GET /api/v1/health`

**请求：** 无

**响应字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | `success` / `not_ready` |
| ready | boolean | 是否可接业务流量 |
| elapsed_time | string | 运行时长，如 `0h 5m 12s` |
| total_requests | int | 累计请求数 |
| memory_mb | float | 进程内存（MB） |
| gpu | object | GPU 配置与解析后的 YOLO 设备 |
| screen_model | object | YOLO 模型加载与 warmup 状态 |

**screen_model 子字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| loaded | boolean | 权重是否已加载 |
| warmed_up | boolean | 是否已完成 GPU warmup |
| weights | string | 权重路径 |
| device | int/string | 实际推理设备 |
| device_name | string | GPU 名称（可选） |
| gpu_memory_mb | float | GPU 显存占用（MB，可选） |
| aattn_patched | int | 兼容补丁层数 |

**响应示例：**

```json
{
  "status": "success",
  "ready": true,
  "elapsed_time": "0h 2m 40s",
  "total_requests": 0,
  "memory_mb": 1451.01,
  "gpu": {
    "enabled": true,
    "device_id": "1",
    "require_gpu": true,
    "tilt_inference_device": "cpu",
    "device_id_config": "1",
    "yolo_device_resolved": 1,
    "cuda_visible_devices": null
  },
  "screen_model": {
    "loaded": true,
    "warmed_up": true,
    "weights": "/app/model/screen.pt",
    "device": 1,
    "device_name": "NVIDIA GeForce RTX 4090 D",
    "gpu_memory_mb": 512.0,
    "aattn_patched": 16
  }
}
```

**HTTP 503：** `preload_at_startup=true` 且 YOLO 未 warmup 完成时。

---

## 3. 倾斜检测

**URL：** `POST /detect_tilt` 或 `POST /api/v1/detect_tilt`

**功能：** 基于 OpenCV 边缘与线段分析，估算画面倾斜角度（CPU）。

### 请求方式 A：JSON（推荐）

**Content-Type：** `application/json`

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| images | R | string | 图片 Base64 |
| image | O | string | 与 images 二选一 |
| tilt_threshold | O | float | 倾斜判定阈值（度），默认读 `config.toml` |

```json
{
  "images": "<base64>",
  "tilt_threshold": 1.5
}
```

### 请求方式 B：纯 Base64

**Content-Type：** `text/plain`

**Body：** 直接为 Base64 字符串（无 JSON 包裹）。此方式**不能**传 `tilt_threshold`，使用配置默认值。

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 200 表示成功 |
| start_time | string | 开始时间戳 |
| end_time | string | 结束时间戳 |
| msg | string | 结果描述 |
| tilt_threshold | float | 实际使用的阈值 |
| result | object | 检测结果 |
| result.is_tilted | boolean | `angle > tilt_threshold` 时为 true |
| result.angle | float | 倾斜角度（度，绝对值） |
| result.cost_ms | float | 耗时（毫秒） |

**成功示例：**

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

**失败示例：**

```json
{
  "code": 400,
  "msg": "Missing field \"images\" or \"image\""
}
```

---

## 4. 屏幕类型检测（YOLO）

**URL：** `POST /detect_screen` 或 `POST /api/v1/detect_screen`

**功能：** 使用 `model/screen.pt` 检测屏幕类型（GPU）；仅返回 **label 0–3**。

**Content-Type：** `application/json`（必须）

### 请求参数

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| images | R | string / string[] | 单图 Base64 或多图数组 |
| conf | O | float | 置信度阈值，默认 0.25 |
| iou | O | float | NMS IoU，默认 0.45 |

**单图请求：**

```json
{
  "images": "<base64>",
  "conf": 0.25,
  "iou": 0.45
}
```

**多图请求：**

```json
{
  "images": ["<base64_1>", "<base64_2>"],
  "conf": 0.3
}
```

- 单次最多图片数：默认 **16**（`max_batch_size`）

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 200 |
| start_time | string | 开始时间戳 |
| end_time | string | 结束时间戳 |
| msg | string | 检测完成 / 未识别到有效类型 |
| conf | float | 实际使用的 conf |
| iou | float | 实际使用的 iou |
| total | int | 图片数量 |
| results | array | 每张图的结果 |
| results[].index | int | 序号，从 0 |
| results[].cost_ms | float | 单图耗时（ms） |
| results[].primary | object/null | 置信度最高的框 |
| results[].detections | array | 全部有效检测（label 0–3） |
| results[].primary.label | int | 0 蓝 / 1 黑 / 2 白 / 3 正常 |
| results[].primary.confidence | float | 0~1 |
| results[].primary.box | float[4] | `[x1, y1, x2, y2]` 左上 + 右下，像素 |

**成功示例：**

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
        "box": [936.0, 55.0, 1697.0, 493.0]
      },
      "detections": [
        {
          "label": 3,
          "confidence": 0.926,
          "box": [936.0, 55.0, 1697.0, 493.0]
        }
      ]
    }
  ]
}
```

---

## 5. 配置查询

**URL：** `GET /config` 或 `GET /api/v1/config`

**请求：** 无

**响应：** 返回 `app`、`server`、`gpu`、`detection`、`screen_detection`、`runtime` 配置对象（`config.toml` 快照）。

---

## 6. 配置重载

**URL：** `POST /config/reload` 或 `POST /api/v1/config/reload`

**请求：** 无 Body

**响应示例：**

```json
{
  "code": 200,
  "msg": "Config reloaded",
  "detection": { "tilt_threshold": 1.5, "...": "..." },
  "screen_detection": {
    "weights_path": "model/screen.pt",
    "conf": 0.25,
    "iou": 0.45,
    "allowed_class_ids": [0, 1, 2, 3],
    "max_batch_size": 16,
    "preload_at_startup": true
  }
}
```

**热重载范围：**

| 配置段 | 是否支持热重载 |
|--------|----------------|
| `[detection]` | ✅ |
| `[screen_detection]` 部分字段 | ✅ |
| `[gpu]`、`[server]`（含 workers、port） | ❌ 需重启 |

---

## 错误码汇总

| 错误码 | HTTP | 说明 | 适用接口 |
|--------|------|------|----------|
| 200 | 200 | 成功 | 全部业务接口 |
| 400 | 400 | 参数错误：缺字段、Base64 无效、图片过大、Content-Type 错误等 | detect_tilt、detect_screen |
| 500 | 500 | 服务内部错误 | detect_tilt、detect_screen |
| 503 | 503 | 服务未就绪（YOLO 未 preload/warmup） | health |

---

## 附录：label 枚举与配置项

### 屏幕检测 label

| label | 含义 |
|-------|------|
| 0 | blue-screen（蓝屏） |
| 1 | black-screen（黑屏） |
| 2 | white-screen（白屏） |
| 3 | normal-screen（正常屏） |

### 常用配置项（config.toml）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| server.port | 8880 | 监听端口 |
| server.workers | 1 | Uvicorn worker 数，单 GPU 建议 1 |
| gpu.enabled | true | 是否使用 GPU |
| gpu.device_id | "1" | YOLO 使用的 GPU 编号 |
| gpu.require_gpu | true | 启动时校验 CUDA |
| screen_detection.preload_at_startup | true | 启动时预加载 + warmup |
| screen_detection.conf | 0.25 | 默认置信度 |
| screen_detection.iou | 0.45 | 默认 IoU |
| screen_detection.max_batch_size | 16 | 单次最大图片数 |
| detection.tilt_threshold | 1.5 | 倾斜判定阈值（度） |
| runtime.max_image_bytes | 10485760 | 单图最大 10MB |

### Docker device_id 说明

| docker run | config device_id | 容器内实际设备 |
|------------|------------------|----------------|
| `--gpus all` | `"0"` / `"1"` / `"2"` | 对应 cuda 编号 |
| `--gpus '"device=1"'` | `"1"` | 自动映射为 **cuda:0** |

---

## 接口明细表（填表格式）

> 以下表格列与 Excel 表头一致，可按行复制到接口登记表。

### 表头说明

| 列 | 含义 |
|----|------|
| 版本说明 | 接口版本 |
| 功能类型 | 业务分类 |
| URL地址 说明 | 路径说明 |
| 请求动作 | HTTP 方法 |
| 参数说明【请求】 | 入参名称 |
| 要求与否(R/O) | R=必填，O=可选 |
| 类型类型 | 数据类型 |
| 类型说明 | 参数说明 |
| 参数说明【应答】 | 出参名称 |
| 要求与否 | R/O |
| 类型类型 | 数据类型 |
| 类型说明 | 参数说明 |
| 错误码 | 业务/HTTP 错误码 |
| 错误码说明 | 说明 |
| 样例说明 | 请求/响应示例 |

---

### 3.1 倾斜检测 — 明细行

| 版本说明 | 功能类型 | URL地址 说明 | 请求动作 | 参数说明【请求】 | R/O | 类型 | 类型说明 | 参数说明【应答】 | R/O | 类型 | 类型说明 | 错误码 | 错误码说明 | 样例说明 |
|---------|---------|-------------|---------|----------------|-----|------|---------|----------------|-----|------|---------|------|---------|---------|
| v1.0.0 | 倾斜检测 | POST /detect_tilt 或 /api/v1/detect_tilt | POST | Content-Type | R | string | application/json 或 text/plain | code | R | int | 200=成功 | 400 | 参数错误 | 见 §3 |
| v1.0.0 | 倾斜检测 | 同上 | POST | images | R | string | Base64（JSON） | msg | R | string | 描述 | 500 | 内部错误 | |
| v1.0.0 | 倾斜检测 | 同上 | POST | image | O | string | 同 images | start_time | R | string | 开始时间戳 | | | |
| v1.0.0 | 倾斜检测 | 同上 | POST | tilt_threshold | O | float | 角度阈值(度) | end_time | R | string | 结束时间戳 | | | |
| v1.0.0 | 倾斜检测 | 同上 | POST | Body(整体) | O | string | text/plain 时为纯 Base64 | tilt_threshold | R | float | 实际阈值 | | | |
| v1.0.0 | 倾斜检测 | 同上 | POST | | | | | result.is_tilted | R | boolean | 是否倾斜 | | | |
| v1.0.0 | 倾斜检测 | 同上 | POST | | | | | result.angle | R | float | 角度(度) | | | |
| v1.0.0 | 倾斜检测 | 同上 | POST | | | | | result.cost_ms | R | float | 耗时(ms) | | | |

---

### 4.1 屏幕检测 — 明细行

| 版本说明 | 功能类型 | URL地址 说明 | 请求动作 | 参数说明【请求】 | R/O | 类型 | 类型说明 | 参数说明【应答】 | R/O | 类型 | 类型说明 | 错误码 | 错误码说明 | 样例说明 |
|---------|---------|-------------|---------|----------------|-----|------|---------|----------------|-----|------|---------|------|---------|---------|
| v1.0.0 | 屏幕检测 | POST /detect_screen 或 /api/v1/detect_screen | POST | Content-Type | R | string | application/json | code | R | int | 200 | 400 | 参数错误 | 见 §4 |
| v1.0.0 | 屏幕检测 | 同上 | POST | images | R | string/array | Base64 单图或数组 | msg | R | string | 结果描述 | 500 | 内部错误 | |
| v1.0.0 | 屏幕检测 | 同上 | POST | conf | O | float | 置信度 | conf | R | float | 实际 conf | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | iou | O | float | NMS IoU | iou | R | float | 实际 iou | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | | | | | total | R | int | 图片数 | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | | | | | results[].primary.label | R | int | 0~3 | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | | | | | results[].primary.confidence | R | float | 置信度 | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | | | | | results[].primary.box | R | float[4] | x1,y1,x2,y2 | | | |
| v1.0.0 | 屏幕检测 | 同上 | POST | | | | | results[].detections | R | array | 全部有效框 | | | |

---

### 2.1 健康检查 — 明细行

| 版本说明 | 功能类型 | URL地址 说明 | 请求动作 | 参数说明【请求】 | R/O | 类型 | 类型说明 | 参数说明【应答】 | R/O | 类型 | 类型说明 | 错误码 | 错误码说明 | 样例说明 |
|---------|---------|-------------|---------|----------------|-----|------|---------|----------------|-----|------|---------|------|---------|---------|
| v1.0.0 | 健康检查 | GET /health 或 /api/v1/health | GET | 无 | - | - | - | status | R | string | success/not_ready | 503 | 未就绪 | 见 §2 |
| v1.0.0 | 健康检查 | 同上 | GET | | | | | ready | R | boolean | 可接流量 | | | |
| v1.0.0 | 健康检查 | 同上 | GET | | | | | screen_model.loaded | R | boolean | 模型已加载 | | | |
| v1.0.0 | 健康检查 | 同上 | GET | | | | | screen_model.warmed_up | R | boolean | GPU已warmup | | | |

---

*文档版本：v1.0.0 | 更新日期：2026-05-25 | 对应仓库：jy-algorithm-app-screen_det-server*

# 构建阶段：PyArmor 加密混淆（须与运行阶段 Python 版本一致，否则 pyarmor_runtime.so 无法加载）
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime AS obfuscator

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /build

RUN pip install --no-cache-dir pyarmor -i "$PIP_INDEX_URL"

COPY app/ ./app/
RUN pyarmor gen -O /dist -r app/


# 运行阶段：PyTorch 2.6 + CUDA 11.8 预装，兼容 CUDA 11/12 驱动
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# torch 已由基础镜像提供（2.6.0+cu118），见 requirements-docker.txt 说明
RUN pip install --retries 10 --timeout 120 -r requirements-docker.txt -i "$PIP_INDEX_URL" \
    && pip install --retries 10 --timeout 120 "ultralytics>=8.3.120" --no-deps -i "$PIP_INDEX_URL"

COPY --from=obfuscator /dist/app ./app
COPY --from=obfuscator /dist/pyarmor_runtime_000000 ./pyarmor_runtime_000000

COPY model/ ./model/
COPY start.sh ./start.sh

RUN chmod +x start.sh && mkdir -p logs

EXPOSE 8880

CMD ["./start.sh"]

# Jetson AGX Xavier production image. r35.4.1 is NVIDIA's latest published
# JetPack 5 l4t-jetpack image and matches the JetPack 5.1.2 wheel set below.
ARG L4T_VERSION=r35.4.1
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION} AS runtime-base

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates curl libgomp1 libopenblas-base libopenmpi-dev python3-pip \
        software-properties-common \
    && add-apt-repository --yes ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install --no-install-recommends -y libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip==24.3.1 'setuptools<76' wheel

ARG TORCH_WHEEL=torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
ARG TORCH_SHA256=5112e2ef5051f1003ae2ffb545ae596377df66979d62a954d774f18009064dbe
ARG TORCH_URL=https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/${TORCH_WHEEL}
ARG TORCHVISION_WHEEL=torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl
ARG TORCHVISION_SHA256=7c98106b7f0a5561edee07bed877d7d2a293c72edaef59eec7794d4809ef7de2
ARG TORCHVISION_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2%2Bc6f3977-cp38-cp38-linux_aarch64.whl
ARG ORT_WHEEL=onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
ARG ORT_SHA256=f2deb6cb314a8a6f793753fbd9ce5bd73e080ca89553dab5e580ac82a9f1ea07
ARG ORT_URL=https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl

COPY requirements-jetson.txt ./
RUN python3 -m pip install numpy==1.24.4 \
    && curl -fL --retry 3 -o "/tmp/${TORCH_WHEEL}" "${TORCH_URL}" \
    && curl -fL --retry 3 -o "/tmp/${TORCHVISION_WHEEL}" "${TORCHVISION_URL}" \
    && curl -fL --retry 3 -o "/tmp/${ORT_WHEEL}" "${ORT_URL}" \
    && echo "${TORCH_SHA256}  /tmp/${TORCH_WHEEL}" | sha256sum -c - \
    && echo "${TORCHVISION_SHA256}  /tmp/${TORCHVISION_WHEEL}" | sha256sum -c - \
    && echo "${ORT_SHA256}  /tmp/${ORT_WHEEL}" | sha256sum -c - \
    && python3 -m pip install "/tmp/${TORCH_WHEEL}" "/tmp/${TORCHVISION_WHEEL}" "/tmp/${ORT_WHEEL}" \
    && python3 -m pip install -r requirements-jetson.txt \
    && python3 -m pip uninstall -y opencv-python opencv-contrib-python \
    && python3 -m pip install --force-reinstall --no-deps opencv-contrib-python-headless==5.0.0.93 \
    && rm -f "/tmp/${TORCH_WHEEL}" "/tmp/${TORCHVISION_WHEEL}" "/tmp/${ORT_WHEEL}" \
    && python3 -c "import onnxruntime as ort; providers=ort.get_available_providers(); print('onnxruntime providers', providers); assert 'CUDAExecutionProvider' in providers" \
    && python3 -c "import cv2; print('headless cv2', cv2.__version__)"

COPY cartgate ./cartgate

FROM runtime-base AS test

COPY requirements-test.txt ./
RUN python3 -m pip install -r requirements-test.txt
COPY Dockerfile ./
COPY Jenkinsfile ./
COPY pyproject.toml ./
COPY requirements-jetson.txt ./
COPY scripts/deploy_with_rollback.sh scripts/validate_model_bundle.sh ./scripts/
COPY tests ./tests
RUN python3 -m pytest tests -q -p no:cacheprovider

FROM runtime-base AS runtime

ENV CARTGATE_MODEL_DIR=/models
RUN useradd --create-home --uid 10001 cartgate
USER cartgate
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "cartgate.server.api:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

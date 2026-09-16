ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cartgate

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cartgate ./cartgate
RUN pip install --no-cache-dir .

USER cartgate
ENV CARTGATE_MODEL_DIR=/models
EXPOSE 8000

CMD ["uvicorn", "cartgate.server.api:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

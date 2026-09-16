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

FROM ${BASE_IMAGE} AS test
WORKDIR /app
COPY --from=0 /usr/local /usr/local
COPY --from=0 /app/cartgate /app/cartgate
COPY tests ./tests
RUN python -m pytest tests -q -p no:cacheprovider

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=0 /usr/local /usr/local
COPY --from=0 /app/cartgate /app/cartgate

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 cartgate

ENV CARTGATE_MODEL_DIR=/models
USER cartgate
EXPOSE 8000

CMD ["uvicorn", "cartgate.server.api:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 IMAGE ENV_FILE MODEL_HOST_DIR" >&2
    exit 2
fi

image=$1
env_file=$2
model_host_dir=$3
container_name=${CARTGATE_CONTAINER_NAME:-cartgate-ai-server}
network_name=${CARTGATE_DOCKER_NETWORK:-cartAider-network}
rollback_name="${container_name}-rollback-${BUILD_NUMBER:-manual}"
health_attempts=${HEALTH_ATTEMPTS:-24}
health_delay_seconds=${HEALTH_DELAY_SECONDS:-5}
backup_created=0

wait_for_health() {
    target=$1
    attempt=1
    while [ "$attempt" -le "$health_attempts" ]; do
        if docker exec "$target" python3 -c \
            "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)); assert data.get('status') == 'ready'; assert 'CUDAExecutionProvider' in data.get('provider', [])"; then
            return 0
        fi
        if [ "$attempt" -lt "$health_attempts" ]; then
            sleep "$health_delay_seconds"
        fi
        attempt=$((attempt + 1))
    done
    return 1
}

rollback() {
    rc=$1
    trap - 0 1 2 15
    set +e

    if docker container inspect "$container_name" >/dev/null 2>&1; then
        docker logs --tail 200 "$container_name" >&2
        if ! docker rm --force "$container_name" >/dev/null 2>&1; then
            echo "automatic rollback failed: replacement could not be removed; backup remains $rollback_name" >&2
            exit 125
        fi
    fi

    if [ "$backup_created" -eq 1 ]; then
        if ! docker rename "$rollback_name" "$container_name"; then
            echo "automatic rollback failed: backup remains $rollback_name" >&2
            exit 125
        fi
        if ! docker start "$container_name" >/dev/null; then
            echo "automatic rollback failed: previous container is stopped as $container_name" >&2
            exit 125
        fi
        if ! wait_for_health "$container_name"; then
            echo "automatic rollback failed: restored container did not become GPU-ready" >&2
            exit 125
        fi
        echo "restored previous healthy container: $container_name" >&2
    fi
    exit "$rc"
}

trap 'exit 129' 1
trap 'exit 130' 2
trap 'exit 143' 15

if docker container inspect "$rollback_name" >/dev/null 2>&1; then
    echo "refusing deployment: preserved rollback container already exists: $rollback_name" >&2
    exit 1
fi

if docker container inspect "$container_name" >/dev/null 2>&1; then
    if ! docker stop "$container_name" >/dev/null; then
        if ! docker start "$container_name" >/dev/null; then
            echo "failed to stop incumbent cleanly and could not ensure it was running: $container_name" >&2
            exit 125
        fi
        echo "failed to stop incumbent; kept it running without deploying" >&2
        exit 1
    fi
    if ! docker rename "$container_name" "$rollback_name"; then
        if ! docker start "$container_name" >/dev/null; then
            echo "failed to rename incumbent and could not restart it: $container_name" >&2
            exit 125
        fi
        echo "failed to preserve incumbent; restarted it without deploying" >&2
        exit 1
    fi
    backup_created=1
fi

trap 'rollback $?' 0

docker run --detach \
    --runtime nvidia \
    --name "$container_name" \
    --network "$network_name" \
    --network-alias fastapi \
    --env-file "$env_file" \
    --mount "type=bind,src=$model_host_dir,dst=/models,readonly" \
    --restart unless-stopped \
    "$image" >/dev/null

if ! wait_for_health "$container_name"; then
    echo "new container failed GPU readiness check: $container_name" >&2
    exit 1
fi

trap - 0 1 2 15
if [ "$backup_created" -eq 1 ] && ! docker rm --force "$rollback_name" >/dev/null; then
    echo "warning: deployment succeeded but backup cleanup failed: $rollback_name" >&2
fi

echo "deployed healthy GPU container: $container_name ($image)"

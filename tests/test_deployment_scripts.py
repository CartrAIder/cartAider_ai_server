import ast
import json
import os
import subprocess
from pathlib import Path
import re


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_model_bundle.sh"
DEPLOY_SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_with_rollback.sh"
PROJECT_ROOT = Path(__file__).parents[1]


def run_validator(path):
    return subprocess.run(
        ["sh", str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def write_bundle(root):
    (root / "runs" / "detector").mkdir(parents=True)
    (root / "out").mkdir()
    for relative_path in ("dino_arc.onnx", "runs/detector/best.pt", "products.csv", "out/gallery.npz"):
        (root / relative_path).write_bytes(b"test artifact")


def run_deployer(tmp_path, *, initial_container=True, fail_at=""):
    assert DEPLOY_SCRIPT.is_file()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
state_path = os.environ["FAKE_DOCKER_STATE"]
with open(state_path) as src:
    state = json.load(src)
fail_at = os.environ.get("FAKE_DOCKER_FAIL", "")
failures = set(filter(None, fail_at.split(",")))
initial_container = os.environ.get("FAKE_INITIAL_CONTAINER") == "1"

def save():
    with open(state_path, "w") as dst:
        json.dump(state, dst)

if args[:2] == ["container", "inspect"]:
    sys.exit(0 if args[2] in state else 1)
if args[0] == "stop":
    if "stop" in failures: sys.exit(1)
    state[args[1]] = "stopped"; save(); sys.exit(0)
if args[0] == "rename":
    old, new = args[1:3]
    failure = "rename_restore" if "rollback" in old else "rename_old"
    if failure in failures: sys.exit(1)
    if old not in state or new in state: sys.exit(1)
    state[new] = state.pop(old); save(); sys.exit(0)
if args[0] == "run":
    if "run" in failures: sys.exit(1)
    name = args[args.index("--name") + 1]
    state[name] = "running"; save(); print("fake-id"); sys.exit(0)
if args[0] == "exec":
    replacement = any("rollback" in name for name in state) or not initial_container
    if "health" in failures and replacement: sys.exit(1)
    restoring = initial_container and not any("rollback" in name for name in state)
    if "restore_health" in failures and restoring: sys.exit(1)
    sys.exit(0)
if args[0] == "logs": sys.exit(0)
if args[0] == "rm":
    state.pop(args[-1], None); save(); sys.exit(0)
if args[0] == "start":
    if "start_restore" in failures: sys.exit(1)
    if args[1] not in state: sys.exit(1)
    state[args[1]] = "running"; save(); sys.exit(0)
sys.exit(1)
"""
    )
    fake_docker.chmod(0o755)
    env_file = tmp_path / "production.env"
    env_file.write_text("SPRING_BASE_URL=http://spring:8080\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    state_path = tmp_path / "docker-state.json"
    state_path.write_text(json.dumps({"cartgate-ai-server": "running"} if initial_container else {}))
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_DOCKER_STATE": str(state_path),
        "FAKE_DOCKER_FAIL": fail_at,
        "FAKE_INITIAL_CONTAINER": "1" if initial_container else "0",
        "BUILD_NUMBER": "42",
        "HEALTH_ATTEMPTS": "1",
        "HEALTH_DELAY_SECONDS": "0",
    }
    result = subprocess.run(
        ["sh", str(DEPLOY_SCRIPT), "cartgate-ai:42", str(env_file), str(model_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return result, docker_log.read_text(), json.loads(state_path.read_text())


def test_model_bundle_validator_accepts_an_npz_only_bundle(tmp_path):
    """Catches deployment rejecting the portable gallery shipped for production."""
    write_bundle(tmp_path)

    result = run_validator(tmp_path)

    assert result.returncode == 0, result.stderr


def test_model_bundle_validator_rejects_a_pickle_only_bundle(tmp_path):
    """Catches deployment accepting a NumPy-version-dependent pickle gallery."""
    write_bundle(tmp_path)
    (tmp_path / "out/gallery.npz").unlink()
    (tmp_path / "out/gallery.pkl").write_bytes(b"incompatible pickle")

    result = run_validator(tmp_path)

    assert result.returncode != 0
    assert "out/gallery.npz" in result.stderr


def test_model_bundle_validator_rejects_the_outer_models_directory(tmp_path):
    """Catches mounting models/ when the bundle root is models/CartGate_AI/."""
    nested_bundle = tmp_path / "CartGate_AI"
    write_bundle(nested_bundle)

    result = run_validator(tmp_path)

    assert result.returncode != 0
    assert "dino_arc.onnx" in result.stderr


def test_model_bundle_validator_rejects_an_empty_artifact(tmp_path):
    """Catches Jenkins accepting a truncated model file before replacing the server."""
    write_bundle(tmp_path)
    (tmp_path / "dino_arc.onnx").write_bytes(b"")

    result = run_validator(tmp_path)

    assert result.returncode != 0
    assert "dino_arc.onnx" in result.stderr


def test_model_bundle_validator_rejects_a_directory_named_like_an_artifact(tmp_path):
    """Catches a same-named directory satisfying a size-only artifact check."""
    write_bundle(tmp_path)
    (tmp_path / "dino_arc.onnx").unlink()
    (tmp_path / "dino_arc.onnx").mkdir()

    result = run_validator(tmp_path)

    assert result.returncode != 0
    assert "dino_arc.onnx" in result.stderr


def test_production_image_targets_jetpack_5_on_arm64():
    """Catches an x86/Python 3.10+ base making the production image unusable on Xavier."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    assert "nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION}" in dockerfile
    assert "ARG L4T_VERSION=r35.4.1" in dockerfile
    assert "requirements-jetson.txt" in dockerfile
    assert "linux/amd64" not in dockerfile
    assert "python:3.11" not in dockerfile


def test_jetson_runtime_dependencies_do_not_install_cuda_12_packages():
    """Catches PyPI CUDA 12 wheels bypassing the JetPack 5 CUDA 11.4 runtime."""
    requirements_path = PROJECT_ROOT / "requirements-jetson.txt"
    assert requirements_path.is_file()
    requirements = requirements_path.read_text()
    pins = [line.strip() for line in requirements.splitlines() if line.strip() and not line.startswith("#")]

    assert "numpy==1.24.4" in requirements
    assert "fastapi==0.115.8" in requirements
    assert "uvicorn[standard]==0.30.6" in requirements
    assert "python-multipart==0.0.20" in requirements
    assert all("cu12" not in pin for pin in pins)
    assert all(not pin.startswith("onnxruntime-gpu") for pin in pins)


def test_package_metadata_accepts_jetpack_python_38():
    """Catches package installation being rejected by Xavier's system Python."""
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text()
    requires_python = re.search(r'^requires-python\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert requires_python is not None
    assert requires_python.group(1) == ">=3.8"


def test_jenkins_builds_locally_and_enables_the_nvidia_runtime():
    """Catches Jenkins deploying an emulated x86 container without Jetson GPU access."""
    jenkinsfile = (PROJECT_ROOT / "Jenkinsfile").read_text()

    assert "linux/amd64" not in jenkinsfile
    deployer = DEPLOY_SCRIPT.read_text() if DEPLOY_SCRIPT.exists() else ""
    assert "--runtime nvidia" in deployer
    assert "L4T_VERSION" in jenkinsfile


def test_successful_deploy_removes_the_rollback_container(tmp_path):
    """Catches a successful release leaking the stopped previous container."""
    result, docker_log, state = run_deployer(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "stop cartgate-ai-server" in docker_log
    assert "rename cartgate-ai-server cartgate-ai-server-rollback-42" in docker_log
    assert "rm --force cartgate-ai-server-rollback-42" in docker_log
    assert state == {"cartgate-ai-server": "running"}
    assert "CUDAExecutionProvider" in docker_log


def test_failed_health_check_restores_the_previous_container(tmp_path):
    """Catches a bad release leaving the service down instead of restoring the previous image."""
    result, docker_log, state = run_deployer(tmp_path, fail_at="health")

    assert result.returncode != 0
    assert "logs --tail 200 cartgate-ai-server" in docker_log
    assert "rm --force cartgate-ai-server" in docker_log
    assert "rename cartgate-ai-server-rollback-42 cartgate-ai-server" in docker_log
    assert "start cartgate-ai-server" in docker_log
    assert state == {"cartgate-ai-server": "running"}


def test_stop_failure_never_removes_the_incumbent(tmp_path):
    result, docker_log, state = run_deployer(tmp_path, fail_at="stop")

    assert result.returncode != 0
    assert state == {"cartgate-ai-server": "running"}
    assert "rm --force cartgate-ai-server\n" not in docker_log


def test_rename_failure_restarts_and_preserves_the_incumbent(tmp_path):
    result, docker_log, state = run_deployer(tmp_path, fail_at="rename_old")

    assert result.returncode != 0
    assert "start cartgate-ai-server" in docker_log
    assert state == {"cartgate-ai-server": "running"}
    assert "rm --force cartgate-ai-server\n" not in docker_log


def test_run_failure_restores_the_previous_container(tmp_path):
    result, _, state = run_deployer(tmp_path, fail_at="run")

    assert result.returncode != 0
    assert state == {"cartgate-ai-server": "running"}


def test_restore_failure_keeps_the_backup_for_manual_recovery(tmp_path):
    result, _, state = run_deployer(tmp_path, fail_at="health,rename_restore")

    assert result.returncode == 125
    assert state == {"cartgate-ai-server-rollback-42": "stopped"}
    assert "automatic rollback failed" in result.stderr


def test_first_deployment_failure_leaves_no_partial_container(tmp_path):
    result, _, state = run_deployer(tmp_path, initial_container=False, fail_at="health")

    assert result.returncode != 0
    assert state == {}


def test_test_image_contains_deployment_contract_inputs():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()

    for path in ("Dockerfile", "Jenkinsfile", "pyproject.toml", "requirements-jetson.txt"):
        assert f"COPY {path} " in dockerfile
    assert "COPY scripts/deploy_with_rollback.sh" in dockerfile


def test_python_38_modules_postpone_modern_type_annotations():
    """Catches Python 3.9+ annotation evaluation crashing an import on JetPack 5."""
    incompatible = []
    builtin_generics = {"dict", "list", "set", "tuple"}

    for module_path in (PROJECT_ROOT / "cartgate").rglob("*.py"):
        source = module_path.read_text()
        tree = ast.parse(source)
        uses_new_annotations = any(
            (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr))
            or (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in builtin_generics
            )
            for node in ast.walk(tree)
        )
        postpones_annotations = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
            for node in tree.body[:2]
        )
        if uses_new_annotations and not postpones_annotations:
            incompatible.append(str(module_path.relative_to(PROJECT_ROOT)))

    assert incompatible == []


def test_python_38_modules_use_annotated_backport():
    """Catches importing typing.Annotated, which is unavailable on Python 3.8."""
    incompatible = []
    for module_path in (PROJECT_ROOT / "cartgate").rglob("*.py"):
        tree = ast.parse(module_path.read_text())
        if any(
            isinstance(node, ast.ImportFrom)
            and node.module == "typing"
            and any(alias.name == "Annotated" for alias in node.names)
            for node in tree.body
        ):
            incompatible.append(str(module_path.relative_to(PROJECT_ROOT)))

    assert incompatible == []


def test_fastapi_route_annotations_are_runtime_evaluable_on_python_38():
    """Catches FastAPI re-evaluating postponed Python 3.10 annotation syntax."""
    source = (PROJECT_ROOT / "cartgate" / "server" / "api.py").read_text()

    assert "Annotated[list[" not in source
    assert ") -> dict[" not in source

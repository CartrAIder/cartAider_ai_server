import hashlib
import json

import pytest

from cartgate.server.settings import Settings


def test_settings_load_gate_api_key_hashes(monkeypatch):
    expected = {"GATE-01": hashlib.sha256(b"secret").hexdigest()}
    monkeypatch.setenv("GATE_API_KEY_HASHES", json.dumps(expected))

    settings = Settings.from_env()

    assert settings.gate_api_key_hashes == expected


def test_settings_reject_missing_gate_api_key_hashes(monkeypatch):
    monkeypatch.delenv("GATE_API_KEY_HASHES", raising=False)

    with pytest.raises(ValueError, match="GATE_API_KEY_HASHES"):
        Settings.from_env()


@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        "[]",
        "{}",
        '{"": "' + ("a" * 64) + '"}',
        '{"GATE-01": "not-a-sha256-hash"}',
    ],
)
def test_settings_reject_invalid_gate_api_key_hashes(monkeypatch, value):
    monkeypatch.setenv("GATE_API_KEY_HASHES", value)

    with pytest.raises(ValueError, match="GATE_API_KEY_HASHES"):
        Settings.from_env()

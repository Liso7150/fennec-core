"""Shared, read-only JSON configuration. Loading never creates files or starts services."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _path(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


@dataclass(frozen=True)
class Config:
    engine_root: Path
    suite_root: Path
    models_root: Path
    cache_root: Path
    python: str
    model: str
    engine_url: str
    web_root: Path
    workspace: Path
    backend: str = "auto"
    ctx: int = 0
    policy: str = "quality"
    api_key: str = field(default="", repr=False)


def load_config(path: str | Path | None = None, *, environ=None, home: Path | None = None) -> Config:
    env = os.environ if environ is None else environ
    home = Path(home or Path.home()).resolve()
    location = Path(path or env.get("FENNEC_CONFIG") or home / ".fennec" / "config.json").expanduser().resolve()
    data = {}
    if location.is_file():
        try:
            data = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError("Cannot read Fennec configuration JSON") from exc
    elif path or env.get("FENNEC_CONFIG"):
        raise ConfigError("The explicitly selected Fennec configuration does not exist")
    if not isinstance(data, dict):
        raise ConfigError("Fennec configuration must be a JSON object")
    if any(k.lower() in {"api_key", "app_token", "token", "fennec_api_key", "fennec_app_token", "coli_api_key"} for k in data):
        raise ConfigError("Secrets belong in environment variables, not configuration JSON")

    def default(windows, portable):
        p = Path(windows)
        return str(p if os.name == "nt" and p.is_dir() else home / portable)

    values = {
        "engine_root": default("D:/Colibri", "Colibri"),
        "suite_root": default("D:/Fennec-Suite", "Fennec-Suite"),
        "models_root": default("D:/fennec-models", "fennec-models"),
        "cache_root": default("D:/.cache", ".cache/fennec"),
        "model": "", "engine_url": "http://127.0.0.1:8184",
        "backend": "auto", "ctx": 0, "policy": "quality",
    }
    keys = set(values) | {"python", "web_root", "workspace"}
    values.update({k: v for k, v in data.items() if k in keys})
    for variable, key in {
        "FENNEC_HOME": "engine_root", "FENNEC_MODELS": "models_root",
        "FENNEC_CACHE": "cache_root", "FENNEC_PYTHON": "python",
        "FENNEC_MODEL": "model", "FENNEC_ENGINE_URL": "engine_url",
        "FENNEC_WORKSPACE": "workspace",
    }.items():
        if variable in env:
            values[key] = env[variable]
    for key, value in values.items():
        if key != "ctx" and (not isinstance(value, str) or (not value.strip() and key != "model")):
            raise ConfigError(f"{key} must be a nonempty string")
    for key in ("engine_root", "suite_root", "models_root", "cache_root"):
        values[key] = _path(values[key], location.parent)
    values["web_root"] = _path(values.get("web_root", str(values["suite_root"] / "fennec-coding-agent" / "web")), location.parent)
    values["workspace"] = _path(values.get("workspace", str(values["suite_root"])), location.parent)
    if values["model"]:
        values["model"] = str(_path(values["model"], values["models_root"]))
    executable = values.get("python")
    if not executable:
        venv = values["engine_root"] / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        executable = str(venv) if venv.is_file() else sys.executable
    elif not Path(executable).is_absolute():
        executable = shutil.which(executable) or str(_path(executable, location.parent))
    values["python"] = executable
    if values["backend"] not in {"auto", "vulkan", "metal", "cpu"}:
        raise ConfigError("backend must be auto, vulkan, metal or cpu")
    if values["policy"] not in {"quality", "balanced", "experimental-fast"}:
        raise ConfigError("policy must be quality, balanced or experimental-fast")
    if type(values["ctx"]) is not int or not 0 <= values["ctx"] <= 1_048_576:
        raise ConfigError("ctx must be an integer from 0 (automatic) to 1048576")
    try:
        url = urlsplit(values["engine_url"])
        valid = (url.scheme in {"http", "https"} and url.hostname and url.port != 0
                 and not url.username and not url.password and url.path in {"", "/"}
                 and not url.query and not url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError("engine_url must be an HTTP(S) origin without credentials, path or query")
    values["engine_url"] = values["engine_url"].rstrip("/")
    key = env.get("FENNEC_API_KEY") or env.get("COLI_API_KEY")
    if not key and not is_loopback(url.hostname):
        raise ConfigError("A non-loopback engine requires FENNEC_API_KEY or COLI_API_KEY")
    values["api_key"] = key or "local"
    return Config(**values)

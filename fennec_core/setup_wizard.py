"""Headless first-run setup: detect this machine, choose paths, write the shared config.

The same config contract every Fennec app reads (`~/.fennec/config.json` or
`FENNEC_CONFIG`, overridable with `FENNEC_*`). The desktop app has an in-app
wizard; this module is its terminal counterpart for servers and CI:

    fennec setup                 # interactive
    fennec setup --yes           # accept detected defaults
    fennec setup --yes --json    # machine-readable result

Nothing is downloaded, installed or started unless `--install-deps` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_KEYS = {
    "engine_root", "suite_root", "models_root", "cache_root", "python", "model",
    "engine_url", "web_root", "workspace", "backend", "ctx", "policy",
}
SECRET_KEYS = {"api_key", "app_token", "token", "fennec_api_key", "fennec_app_token", "coli_api_key"}
INPUT_KEYS = {
    "root", "engine_root", "suite_root", "models_root", "cache_root", "python",
    "model", "engine_url", "backend", "ctx", "policy",
}
BACKENDS = ("auto", "vulkan", "metal", "cpu")
POLICIES = ("quality", "balanced", "experimental-fast")
ENGINE_FILES = ("c/coli", "c/openai_server.py")
CORE_DEPS = ("fastapi", "uvicorn", "psutil", "pydantic")
MIN_PYTHON = (3, 10)
PYTHON_PROBE = (
    "import importlib.util, json, sys\n"
    "print(json.dumps({'version': '%d.%d.%d' % sys.version_info[:3],"
    "'venv': sys.prefix != sys.base_prefix,"
    "'executable': sys.executable,"
    "'pip': importlib.util.find_spec('pip') is not None}))"
)
VULKAN_LINUX = (
    "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
    "/usr/lib64/libvulkan.so.1",
    "/usr/lib/libvulkan.so.1",
)
UPSTREAM = ("engine base idea JustVugg/colibri (https://github.com/JustVugg/colibri), "
            "fine-tuning lyogavin/airllm (https://github.com/lyogavin/airllm)")


def config_path(environ=None, home=None) -> Path:
    env = os.environ if environ is None else environ
    base = Path(home or Path.home()).expanduser()
    return Path(env.get("FENNEC_CONFIG") or base / ".fennec" / "config.json").expanduser()


def _abs(value, home: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = home / path
    return str(path.absolute())


def default_roots(home=None) -> dict:
    base = Path(home or Path.home()).expanduser()
    root = base / "Fennec"
    return {
        "root": str(root),
        "engine_root": str(root / "engine"),
        "suite_root": str(root / "suite"),
        "models_root": str(root / "models"),
        "cache_root": str(root / "cache"),
    }


def _path_entries(environ) -> list[str]:
    return [entry for entry in (environ.get("PATH") or "").split(os.pathsep) if entry]


def known_python_paths(home: Path, system: str, environ) -> list[Path]:
    if system == "Windows":
        local = Path(environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
        paths = []
        for minor in ("311", "312", "313", "314"):
            paths.append(local / "Programs" / "Python" / f"Python{minor}" / "python.exe")
            paths.append(Path("C:/") / f"Python{minor}" / "python.exe")
        return paths
    if system == "Darwin":
        return [Path("/opt/homebrew/bin/python3.13"), Path("/opt/homebrew/bin/python3.12"),
                Path("/opt/homebrew/bin/python3"), Path("/usr/local/bin/python3.13"),
                Path("/usr/local/bin/python3.12"), Path("/usr/local/bin/python3"),
                Path("/usr/bin/python3")]
    return [Path("/usr/bin/python3.12"), Path("/usr/bin/python3"),
            Path("/usr/local/bin/python3.12"), Path("/usr/local/bin/python3")]


def python_candidates(*, environ=None, home=None, system=None, exists=None) -> list[dict]:
    """Ordered interpreters: FENNEC_PYTHON, PATH, known locations, `py -3`."""
    env = os.environ if environ is None else environ
    base = Path(home or Path.home()).expanduser()
    system = system or platform_module.system()
    exists = exists or (lambda path: Path(path).exists())
    found: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def add(command, args=(), source="PATH"):
        key = (str(command), tuple(args))
        if key in seen:
            return
        seen.add(key)
        found.append({"command": str(command), "args": list(args), "source": source})

    forced = env.get("FENNEC_PYTHON")
    if forced:
        add(Path(forced).expanduser(), (), "FENNEC_PYTHON")
    names = ("python.exe", "python3.exe") if system == "Windows" else ("python3", "python")
    for directory in _path_entries(env):
        for name in names:
            candidate = Path(directory) / name
            if exists(candidate):
                add(candidate)
    for candidate in known_python_paths(base, system, env):
        if exists(candidate):
            add(candidate, source="known location")
    if system == "Windows":
        add("py", ("-3",), "py launcher")
    return found


def probe_python(command, args=(), runner=subprocess.run) -> dict:
    argv = [str(command), *[str(value) for value in args], "-c", PYTHON_PROBE]
    try:
        result = runner(argv, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "command": str(command), "error": str(exc)}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "python did not start").strip()
        return {"ok": False, "command": str(command), "error": detail[:800]}
    try:
        data = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "command": str(command), "error": "python did not report its version"}
    parts = [int(part) for part in str(data.get("version", "0.0.0")).split(".")[:3]]
    parts += [0] * (3 - len(parts))
    major, minor, patch = parts[:3]
    ok = (major, minor) >= MIN_PYTHON
    return {
        "ok": ok,
        "version": data.get("version"),
        "major": major,
        "minor": minor,
        "patch": patch,
        "venv": bool(data.get("venv")),
        "executable": data.get("executable", ""),
        "has_pip": bool(data.get("pip")),
        "command": str(command),
        "args": list(args),
        "error": None if ok else f"Python {data.get('version')} is older than 3.10",
    }


def first_python(*, environ=None, home=None, system=None, runner=subprocess.run) -> dict:
    tried = []
    for candidate in python_candidates(environ=environ, home=home, system=system):
        tried.append(candidate["command"])
        info = probe_python(candidate["command"], candidate["args"], runner=runner)
        if info["ok"]:
            info["source"] = candidate["source"]
            info["tried"] = tried
            return info
    return {"ok": False, "tried": tried,
            "error": "No Python 3.10+ interpreter was found; install it or pass --python"}


def dependency_probe_args(modules=CORE_DEPS) -> list[str]:
    code = (
        "import importlib, json\n"
        "missing = []\n"
        f"for name in {json.dumps(list(modules))}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print(json.dumps({'missing': missing}))"
    )
    return ["-c", code]


def probe_dependencies(python: str, runner=subprocess.run, modules=CORE_DEPS) -> dict:
    try:
        result = runner([python, *dependency_probe_args(modules)],
                        capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"missing": list(modules), "installed": [], "error": str(exc)}
    if result.returncode != 0:
        return {"missing": list(modules), "installed": [],
                "error": (result.stderr or result.stdout or "dependency probe failed").strip()[:800]}
    try:
        data = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"missing": list(modules), "installed": [], "error": "dependency probe returned no JSON"}
    missing = [name for name in data.get("missing", []) if name in modules]
    return {"missing": missing, "installed": [name for name in modules if name not in missing]}


def pip_install_args(python: str, packages=CORE_DEPS, *, venv: bool = False) -> list[str]:
    specs = [str(spec).strip() for spec in packages if str(spec).strip()]
    if not specs:
        raise ValueError("No packages selected")
    if any(any(character in spec for character in ' \t;&|`$"\'') for spec in specs):
        raise ValueError("Invalid package specification")
    return [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
            *([] if venv else ["--user"]), *specs]


def install_dependencies(python: str, *, venv: bool = False, packages=CORE_DEPS,
                         runner=subprocess.run, out=print) -> bool:
    argv = pip_install_args(python, packages, venv=venv)
    out("Installing: " + " ".join(argv))
    try:
        result = runner(argv, capture_output=True, text=True, timeout=30 * 60)
    except (OSError, subprocess.SubprocessError) as exc:
        out(f"pip failed: {exc}")
        return False
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        out(output.strip()[-4000:])
    return result.returncode == 0


def validate_engine(directory) -> dict:
    path = Path(str(directory or "")).expanduser()
    is_directory = path.is_dir()
    missing = [relative for relative in ENGINE_FILES if not (path / relative).is_file()]
    return {"ok": is_directory and not missing, "path": str(path),
            "missing": missing if is_directory else list(ENGINE_FILES),
            "is_directory": is_directory}


def scan_models(directory, *, limit: int = 40) -> list[dict]:
    root = Path(str(directory or "")).expanduser()
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []
    models = []
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            files = [item for item in entry.iterdir() if item.is_file()]
        except OSError:
            continue
        has_config = (entry / "config.json").is_file()
        weights = [item for item in files if item.suffix.lower() in {".safetensors", ".gguf"}]
        if not has_config and not weights:
            continue
        size = 0
        for weight in weights:
            try:
                size += weight.stat().st_size
            except OSError:
                pass
        models.append({
            "name": entry.name,
            "path": str(entry),
            "size_gb": round(size / 1e9, 2),
            "shards": len(weights),
            "has_config": has_config,
            "has_weights": bool(weights),
            "ready": has_config and bool(weights),
        })
    models.sort(key=lambda model: (not model["ready"], model["name"].lower()))
    return models[:limit]


def has_vulkan_loader(system=None, environ=None) -> bool:
    system = system or platform_module.system()
    env = os.environ if environ is None else environ
    if system == "Darwin":
        return False
    if system == "Windows":
        root = Path(env.get("SystemRoot") or env.get("SYSTEMROOT") or "C:/Windows")
        return (root / "System32" / "vulkan-1.dll").exists()
    return any(Path(candidate).exists() for candidate in VULKAN_LINUX)


def _ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (AttributeError, ValueError, OSError):
        return 0.0


def _free_bytes(directory) -> int:
    path = Path(str(directory)).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return int(shutil.disk_usage(str(path)).free)
    except OSError:
        return 0


def device_info(root, *, system=None, environ=None) -> dict:
    system = system or platform_module.system()
    return {
        "os": system,
        "os_version": platform_module.release(),
        "arch": platform_module.machine(),
        "cores": os.cpu_count() or 0,
        "ram_gb": _ram_gb(),
        "free_gb": round(_free_bytes(root) / 1e9, 1),
        "vulkan": has_vulkan_loader(system, environ),
        "metal": system == "Darwin",
    }


def validate_config(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    for key in data:
        if key.lower() in SECRET_KEYS:
            raise ValueError("Secrets belong in FENNEC_API_KEY, not configuration JSON")
        if key not in CONFIG_KEYS:
            raise ValueError(f"Unsupported configuration key: {key}")
    for key, value in data.items():
        if key == "ctx":
            if type(value) is not int or not 0 <= value <= 1_048_576:
                raise ValueError("ctx must be an integer from 0 (automatic) to 1048576")
        elif not isinstance(value, str) or (not value.strip() and key != "model"):
            raise ValueError(f"{key} must be a nonempty string")
    if data.get("backend", "auto") not in BACKENDS:
        raise ValueError("backend must be auto, vulkan, metal or cpu")
    if data.get("policy", "quality") not in POLICIES:
        raise ValueError("policy must be quality, balanced or experimental-fast")
    try:
        url = urlsplit(data.get("engine_url", "http://127.0.0.1:8184"))
        valid = (url.scheme in {"http", "https"} and url.hostname and url.port != 0
                 and not url.username and not url.password
                 and url.path in {"", "/"} and not url.query and not url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("engine_url must be an HTTP(S) origin without credentials, path or query")
    return data


def build_config(values, *, home=None, system=None) -> dict:
    """The same 12 keys the desktop wizard writes, derived from a chosen root."""
    home_path = Path(home or Path.home()).expanduser()
    input_values = dict(values or {})
    for key in input_values:
        if key.lower() in SECRET_KEYS:
            raise ValueError("Secrets belong in FENNEC_API_KEY, not configuration JSON")
        if key not in INPUT_KEYS:
            raise ValueError(f"Unsupported setup option: {key}")
    root = _abs(input_values.get("root") or home_path / "Fennec", home_path)
    root_path = Path(root)
    suite = _abs(input_values.get("suite_root") or root_path / "suite", home_path)
    models = _abs(input_values.get("models_root") or root_path / "models", home_path)
    engine = _abs(input_values.get("engine_root") or root_path / "engine", home_path)
    cache = _abs(input_values.get("cache_root") or root_path / "cache", home_path)
    model = str(input_values.get("model") or "").strip()
    if model and not Path(model).expanduser().is_absolute():
        model = str((Path(models) / model).absolute())
    config = {
        "engine_root": engine,
        "suite_root": suite,
        "models_root": models,
        "cache_root": cache,
        "engine_url": str(input_values.get("engine_url") or "http://127.0.0.1:8184"),
        "workspace": suite,
        "web_root": str(Path(suite) / "fennec-coding-agent" / "web"),
        "backend": input_values.get("backend") or "auto",
        "ctx": input_values.get("ctx") if input_values.get("ctx") is not None else 0,
        "policy": input_values.get("policy") or "quality",
        "model": str(Path(model).absolute()) if model else "",
    }
    if input_values.get("python"):
        config["python"] = _abs(input_values["python"], home_path)
    config["ctx"] = int(config["ctx"])
    return validate_config(config)


def read_config(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_config(path, data, *, backup: bool = True) -> dict:
    validate_config(data)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if destination.exists():
        if not backup:
            raise ValueError(f"Configuration already exists: {destination}")
        backup_path = destination.with_name(destination.name + ".bak-fennec-setup")
        shutil.copy2(destination, backup_path)
    handle_path = None
    try:
        descriptor, handle_path = tempfile.mkstemp(prefix=f".{destination.name}.",
                                                   suffix=".tmp", dir=str(destination.parent))
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(handle_path, 0o600)
        os.replace(handle_path, destination)
        handle_path = None
    finally:
        if handle_path and os.path.exists(handle_path):
            os.unlink(handle_path)
    return {"path": str(destination), "backup": str(backup_path) if backup_path else None}


def _pick_directory(explicit, key, candidates, existing) -> str:
    if explicit:
        return str(Path(explicit).expanduser())
    if existing.get(key):
        return str(existing[key])
    for candidate in candidates:
        if Path(candidate).is_dir():
            return str(candidate)
    return str(candidates[0])


def resolve_values(args, existing: dict, *, home: Path, device: dict) -> dict:
    roots = default_roots(home)
    root = _abs(args.root or roots["root"], home)
    root_path = Path(root)
    models_root = _pick_directory(args.models_root, "models_root",
                                  [root_path / "models", home / "fennec-models"], existing)
    values = {
        "root": root,
        "engine_root": _pick_directory(args.engine_root, "engine_root",
                                       [root_path / "engine", home / "Colibri"], existing),
        "suite_root": _pick_directory(args.suite_root, "suite_root",
                                      [root_path / "suite", home / "Fennec-Suite"], existing),
        "models_root": models_root,
        "cache_root": _pick_directory(args.cache_root, "cache_root",
                                      [root_path / "cache", home / ".cache" / "fennec"], existing),
        "python": str(Path(args.python).expanduser()) if args.python else existing.get("python", ""),
        "backend": args.backend or existing.get("backend") or "auto",
        "policy": args.policy or existing.get("policy") or "quality",
        "ctx": args.ctx if args.ctx is not None else existing.get("ctx",
                                                                 4096 if device["ram_gb"] >= 16 else 2048),
    }
    model = args.model or existing.get("model") or ""
    if not model:
        ready = [entry for entry in scan_models(models_root) if entry["ready"]]
        if len(ready) == 1:
            model = ready[0]["path"]
    values["model"] = model
    return values


def _ask(input_fn, label: str, default) -> str:
    suffix = f" [{default}]" if str(default) else ""
    answer = input_fn(f"{label}{suffix}: ")
    return (answer or "").strip() or str(default)


def prompt_values(values: dict, models: list[dict], device: dict, input_fn, out=print) -> dict:
    updated = dict(values)
    updated["root"] = _ask(input_fn, "Install root", values["root"])
    updated["engine_root"] = _ask(input_fn, "Engine checkout", values["engine_root"])
    updated["models_root"] = _ask(input_fn, "Models folder", values["models_root"])
    updated["cache_root"] = _ask(input_fn, "Cache folder", values["cache_root"])
    updated["python"] = _ask(input_fn, "Python interpreter", values.get("python") or "")
    if models:
        out("Models found:")
        for index, model in enumerate(models, start=1):
            state = "ready" if model["ready"] else "incomplete"
            out(f"  {index}. {model['name']} ({model['size_gb']} GB, {state})")
    current = values.get("model") or ""
    default_index = next((str(index) for index, model in enumerate(models, start=1)
                          if model["path"] == current), "none")
    answer = _ask(input_fn, "Model (number, path, or 'none')", default_index)
    if answer.isdigit() and 1 <= int(answer) <= len(models):
        updated["model"] = models[int(answer) - 1]["path"]
    elif answer.lower() in {"none", "0", "skip"}:
        updated["model"] = ""
    else:
        updated["model"] = answer
    updated["backend"] = _ask(input_fn, f"Backend ({'/'.join(BACKENDS)})", values["backend"])
    try:
        updated["ctx"] = int(_ask(input_fn, "Context tokens (0 = automatic)", values["ctx"]))
    except ValueError:
        updated["ctx"] = values["ctx"]
    updated["policy"] = _ask(input_fn, f"Policy ({'/'.join(POLICIES)})", values["policy"])
    return updated


def _summary(config: dict, python: dict, dependencies: dict, device: dict) -> list[str]:
    lines = [
        f"engine    {config['engine_root']}",
        f"suite     {config['suite_root']}",
        f"models    {config['models_root']}",
        f"cache     {config['cache_root']}",
        f"python    {config.get('python', '(PATH)')}"
        + (f" ({python.get('version')})" if python.get("version") else ""),
        f"model     {config.get('model') or '(none yet)'}",
        f"backend   {config['backend']} · ctx {config['ctx'] if config['ctx'] else 'automatic'} · {config['policy']}",
    ]
    if dependencies.get("missing"):
        lines.append("deps      missing: " + ", ".join(dependencies["missing"]))
    else:
        lines.append("deps      ok")
    lines.append(f"device    {device['os']}/{device['arch']} · {device['cores']} cores · "
                 f"{device['ram_gb']} GB RAM · {device['free_gb']} GB free")
    return lines


def main(argv=None, *, config=None, home=None, environ=None, input_fn=None,
         runner=subprocess.run, out=print, err=None) -> int:
    warn = err or (lambda message: print(message, file=sys.stderr))
    parser = argparse.ArgumentParser(
        prog="fennec setup",
        description="Detect Python, the engine and models, then write the shared Fennec config.")
    parser.add_argument("--root", help="install root (default: ~/Fennec)")
    parser.add_argument("--engine-root", help="Colibri checkout containing c/coli")
    parser.add_argument("--suite-root", help="Fennec suite checkout root")
    parser.add_argument("--models-root", help="model directory")
    parser.add_argument("--cache-root", help="cache directory")
    parser.add_argument("--python", help="absolute Python 3.10+ interpreter")
    parser.add_argument("--model", help="model directory or a name under --models-root")
    parser.add_argument("--backend", choices=BACKENDS, help="engine backend")
    parser.add_argument("--ctx", type=int, help="context length; 0 means automatic")
    parser.add_argument("--policy", choices=POLICIES, help="engine resource policy")
    parser.add_argument("--yes", action="store_true", help="accept detected defaults without prompts")
    parser.add_argument("--dry-run", action="store_true", help="print the config without writing it")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--install-deps", action="store_true",
                        help="install missing Python dependencies with pip before saving")
    args = parser.parse_args(argv)

    env = os.environ if environ is None else environ
    home_path = Path(home or Path.home()).expanduser()
    system = platform_module.system()
    destination = config_path(env, home_path)
    existing = read_config(destination) or dict(config or {})
    roots = default_roots(home_path)
    device = device_info(args.root or existing.get("engine_root") or roots["root"],
                         system=system, environ=env)
    values = resolve_values(args, existing, home=home_path, device=device)

    python = {"ok": False}
    if values.get("python"):
        python = probe_python(values["python"], runner=runner)
    if not python.get("ok"):
        detected = first_python(environ=env, home=home_path, system=system, runner=runner)
        if detected.get("ok"):
            python = detected
            values["python"] = detected["command"]
        else:
            python = detected

    if python.get("ok"):
        values["python"] = python["command"]
    models = scan_models(values["models_root"])
    if not args.yes and not args.dry_run:
        values = prompt_values(values, models, device, input_fn or input, out=out)
        if values.get("python") and values["python"] != python.get("command"):
            python = probe_python(values["python"], runner=runner)

    try:
        config_values = build_config(values, home=home_path, system=system)
    except (TypeError, ValueError) as exc:
        warn(f"fennec setup: {exc}")
        return 1
    engine = validate_engine(config_values["engine_root"])
    if not engine["ok"]:
        message = (f"engine checkout is missing c/coli and c/openai_server.py: "
                   f"{config_values['engine_root']}")
        if not args.dry_run:
            warn(f"fennec setup: {message}. Run fennec-installer or pass --engine-root.")
            return 2
        warn(f"fennec setup: warning: {message}")
    if not python.get("ok"):
        message = python.get("error") or "No Python 3.10+ interpreter was found"
        if not args.dry_run:
            warn(f"fennec setup: {message}. Install Python or pass --python.")
            return 2
        warn(f"fennec setup: warning: {message}")
    dependencies = {"missing": [], "installed": []}
    if python.get("ok"):
        dependencies = probe_dependencies(python["command"], runner=runner)
        if dependencies.get("error"):
            warn(f"fennec setup: dependency probe failed: {dependencies['error']}")
        if dependencies.get("missing") and args.install_deps:
            installed = install_dependencies(python["command"], venv=bool(python.get("venv")),
                                             runner=runner, out=out)
            dependencies = probe_dependencies(python["command"], runner=runner)
            if not installed or dependencies.get("missing"):
                warn("fennec setup: dependencies are still missing: "
                     + ", ".join(dependencies.get("missing", [])))
                return 1

    if args.dry_run:
        if args.json:
            out(json.dumps(config_values, indent=2))
        else:
            for line in _summary(config_values, python, dependencies, device):
                out(line)
            out(f"config    {destination} (not written: --dry-run)")
        return 0

    if not args.yes:
        out("")
        for line in _summary(config_values, python, dependencies, device):
            out(line)
        answer = (input_fn or input)(f"Write {destination}? [Y/n]: ").strip().lower()
        if answer.startswith("n"):
            out("Nothing written.")
            return 1

    try:
        result = write_config(destination, config_values)
    except (OSError, ValueError) as exc:
        warn(f"fennec setup: {exc}")
        return 1
    if args.json:
        out(json.dumps({"written": result["path"], "backup": result["backup"],
                        "config": config_values}, indent=2))
    else:
        out(f"Wrote {result['path']}")
        if result["backup"]:
            out(f"Backed up the previous file to {result['backup']}")
        for line in _summary(config_values, python, dependencies, device):
            out(line)
        out("Next: `fennec diagnose`, then `fennec serve --model <dir>`.")
        out(f"Upstream: {UPSTREAM} - neither is pulled by this command.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

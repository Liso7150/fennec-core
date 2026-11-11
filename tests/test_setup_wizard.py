import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fennec_core import setup_wizard as wizard


def make_engine(root: Path) -> Path:
    (root / "c").mkdir(parents=True)
    (root / "c" / "coli").write_text("", encoding="utf-8")
    (root / "c" / "openai_server.py").write_text("", encoding="utf-8")
    return root


def make_model(directory: Path, name="qwen36-35b", weights=True) -> Path:
    model = directory / name
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (model / "model-00001.safetensors").write_bytes(b"x" * 2048)
    return model


class BuildConfigTests(unittest.TestCase):
    def test_writes_the_same_twelve_keys_as_the_desktop_wizard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Fennec"
            config = wizard.build_config({
                "root": str(root), "python": sys.executable, "backend": "vulkan", "ctx": 8192,
            }, home=directory, system="Linux")
        self.assertEqual(set(config), {
            "engine_root", "suite_root", "models_root", "cache_root", "engine_url",
            "workspace", "web_root", "backend", "ctx", "policy", "python", "model",
        })
        self.assertEqual(config["engine_root"], str(root / "engine"))
        self.assertEqual(config["models_root"], str(root / "models"))
        self.assertEqual(config["cache_root"], str(root / "cache"))
        self.assertEqual(config["suite_root"], str(root / "suite"))
        self.assertEqual(config["web_root"], str(root / "suite" / "fennec-coding-agent" / "web"))
        self.assertEqual(config["workspace"], config["suite_root"])
        self.assertEqual(config["engine_url"], "http://127.0.0.1:8184")
        self.assertEqual(config["model"], "")
        self.assertEqual(config["backend"], "vulkan")
        self.assertEqual(config["ctx"], 8192)

    def test_relative_model_resolves_under_the_models_root(self):
        with tempfile.TemporaryDirectory() as directory:
            config = wizard.build_config({"root": directory, "model": "qwen36-35b"}, home=directory)
            self.assertEqual(Path(config["model"]), Path(directory) / "models" / "qwen36-35b")

    def test_rejects_secrets_and_invalid_enums(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Secrets"):
                wizard.build_config({"api_key": "x"}, home=directory)
            with self.assertRaisesRegex(ValueError, "backend"):
                wizard.build_config({"backend": "npu"}, home=directory)
            with self.assertRaisesRegex(ValueError, "policy"):
                wizard.build_config({"policy": "fastest"}, home=directory)
            with self.assertRaises(ValueError):
                wizard.build_config({"ctx": "not-a-number"}, home=directory)


class ProbeTests(unittest.TestCase):
    def test_python_probe_accepts_310_and_reports_venv_and_pip(self):
        done = subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "version": "3.12.4", "venv": True, "executable": "/opt/py", "pip": True,
        }) + "\n", stderr="")
        info = wizard.probe_python("/opt/py", runner=lambda *a, **k: done)
        self.assertTrue(info["ok"])
        self.assertEqual((info["major"], info["minor"]), (3, 12))
        self.assertTrue(info["venv"])
        self.assertTrue(info["has_pip"])

    def test_python_probe_rejects_old_and_broken_interpreters(self):
        old = subprocess.CompletedProcess([], 0, stdout='{"version": "3.9.18"}\n', stderr="")
        self.assertFalse(wizard.probe_python("py", runner=lambda *a, **k: old)["ok"])
        broken = subprocess.CompletedProcess([], 1, stdout="", stderr="not found")
        result = wizard.probe_python("py", runner=lambda *a, **k: broken)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_candidates_prefer_fennec_python_then_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            forced = base / "custom" / "python3"
            first = base / "py" / "python3"
            second = base / "bin" / "python3"
            wanted = {str(forced), str(first), str(second)}
            candidates = wizard.python_candidates(
                environ={"PATH": os.pathsep.join([str(first.parent), str(second.parent)]),
                         "FENNEC_PYTHON": str(forced)},
                home=base, system="Linux", exists=lambda path: str(path) in wanted)
        self.assertEqual([entry["command"] for entry in candidates],
                         [str(forced), str(first), str(second)])
        self.assertEqual(candidates[0]["source"], "FENNEC_PYTHON")

    def test_windows_candidates_expand_python_exe_and_keep_the_py_launcher(self):
        candidates = wizard.python_candidates(
            environ={"PATH": "C:\\Python", "LOCALAPPDATA": "C:/Users/dev/AppData/Local"},
            home="C:/Users/dev", system="Windows",
            exists=lambda path: str(path) in {"C:\\Python\\python.exe"})
        self.assertEqual(candidates[0]["command"], "C:\\Python\\python.exe")
        self.assertTrue(any(entry["command"] == "py" and entry["args"] == ["-3"] for entry in candidates))

    def test_dependency_probe_reports_missing_modules(self):
        done = subprocess.CompletedProcess([], 0, stdout='{"missing": ["psutil"]}\n', stderr="")
        result = wizard.probe_dependencies("/opt/py", runner=lambda *a, **k: done)
        self.assertEqual(result["missing"], ["psutil"])
        self.assertIn("fastapi", result["installed"])

    def test_pip_args_use_user_scope_outside_a_venv(self):
        argv = wizard.pip_install_args("/opt/py", ["fastapi>=0.110", "pydantic"])
        self.assertEqual(argv[:6], ["/opt/py", "-m", "pip", "install",
                                    "--disable-pip-version-check", "--no-input"])
        self.assertIn("--user", argv)
        self.assertNotIn("--user", wizard.pip_install_args("/opt/venv/python", ["fastapi"], venv=True))
        with self.assertRaisesRegex(ValueError, "Invalid package"):
            wizard.pip_install_args("/opt/py", ["fastapi; rm -rf /"])


class FilesystemTests(unittest.TestCase):
    def test_engine_requires_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            self.assertFalse(wizard.validate_engine(engine)["ok"])
            (engine / "c").mkdir(parents=True)
            (engine / "c" / "coli").write_text("", encoding="utf-8")
            self.assertEqual(wizard.validate_engine(engine)["missing"], ["c/openai_server.py"])
            (engine / "c" / "openai_server.py").write_text("", encoding="utf-8")
            self.assertTrue(wizard.validate_engine(engine)["ok"])

    def test_scan_models_reports_size_and_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_model(root, "ready")
            make_model(root, "partial", weights=False)
            (root / "not-a-model").mkdir()
            models = wizard.scan_models(root)
        self.assertEqual([model["name"] for model in models], ["ready", "partial"])
        ready = next(model for model in models if model["name"] == "ready")
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["size_gb"], 0.0)
        self.assertEqual(ready["shards"], 1)

    def test_atomic_write_backs_up_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / ".fennec" / "config.json"
            first = wizard.build_config({"root": directory}, home=directory)
            wizard.write_config(destination, first)
            second = dict(first, backend="cpu")
            result = wizard.write_config(destination, second)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["backend"], "cpu")
            self.assertTrue(result["backup"] and Path(result["backup"]).is_file())
            self.assertEqual([name for name in os.listdir(destination.parent) if name.endswith(".tmp")], [])
            if os.name != "nt":
                self.assertEqual(destination.stat().st_mode & 0o777, 0o600)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.engine = make_engine(self.root / "Fennec" / "engine")
        self.models = self.root / "Fennec" / "models"
        self.model = make_model(self.models)
        self.config_file = self.home / ".fennec" / "config.json"

    def run_main(self, *arguments, input_fn=None):
        out, err = io.StringIO(), io.StringIO()
        dependencies = {"missing": [], "installed": list(wizard.CORE_DEPS)}
        with mock.patch.object(wizard, "probe_dependencies", return_value=dependencies):
            code = wizard.main(
                ["--root", str(self.root / "Fennec"), "--engine-root", str(self.engine),
                 "--models-root", str(self.models), "--python", sys.executable, *arguments],
                home=self.home, environ={}, input_fn=input_fn, out=out.write, err=err.write)
        return code, out.getvalue(), err.getvalue()

    def test_yes_writes_the_shared_config_and_picks_the_single_model(self):
        code, out, err = self.run_main("--yes", "--json")
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(result["written"], str(self.config_file))
        config = json.loads(self.config_file.read_text(encoding="utf-8"))
        self.assertEqual(config["model"], str(self.model))
        self.assertEqual(config["python"], sys.executable)
        self.assertEqual(config["engine_root"], str(self.engine))
        self.assertEqual(config["backend"], "auto")

    def test_yes_refuses_a_missing_engine_and_writes_nothing(self):
        out, err = io.StringIO(), io.StringIO()
        code = wizard.main(
            ["--root", str(self.root / "absent"), "--python", sys.executable, "--yes"],
            home=self.home, environ={}, out=out.write, err=err.write)
        self.assertEqual(code, 2)
        self.assertIn("engine checkout", err.getvalue())
        self.assertFalse(self.config_file.exists())

    def test_dry_run_prints_without_writing(self):
        code, out, err = self.run_main("--dry-run", "--json")
        self.assertEqual(code, 0, err)
        printed = json.loads(out)
        self.assertEqual(printed["engine_root"], str(self.engine))
        self.assertFalse(self.config_file.exists())

    def test_interactive_blanks_accept_defaults_and_write(self):
        code, out, err = self.run_main(input_fn=lambda prompt: "")
        self.assertEqual(code, 0, err)
        self.assertTrue(self.config_file.is_file())
        self.assertIn("Wrote", out)

    def test_existing_config_is_backed_up_when_rewritten(self):
        existing = wizard.build_config({"root": str(self.root / "Fennec"),
                                        "engine_root": str(self.engine)}, home=self.home)
        wizard.write_config(self.config_file, existing)
        code, out, err = self.run_main("--yes")
        self.assertEqual(code, 0, err)
        backup = Path(str(self.config_file) + ".bak-fennec-setup")
        self.assertTrue(backup.is_file())
        self.assertIn("Backed up", out)

    def test_device_info_has_the_fields_the_summary_uses(self):
        info = wizard.device_info(self.root, environ={})
        for key in ("os", "arch", "cores", "ram_gb", "free_gb", "vulkan", "metal"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()

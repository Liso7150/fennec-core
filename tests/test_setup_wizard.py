import hermetic  # noqa: F401

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup_wizard as wizard


def make_engine(root: Path) -> Path:
    binary = root / "target" / "release" / ("fennec.exe" if os.name == "nt" else "fennec")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    return root


def make_model(directory: Path, name="qwen36-35b", weights=True) -> Path:
    model = directory / name
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (model / "model-00001.safetensors").write_bytes(b"x" * 2048)
    return model


def make_recommended_model(directory: Path) -> Path:
    model = directory / wizard.RECOMMENDED_MODEL_DIR
    model.mkdir(parents=True)
    for name in wizard.RECOMMENDED_MODEL_FILES:
        (model / name).write_bytes(b"x" * 2048)
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

    def test_a_browser_interpreter_pin_survives_the_wizard(self):
        with tempfile.TemporaryDirectory() as directory:
            config = wizard.build_config({"root": directory,
                                          "web_python": "browsenv/python"}, home=directory)
            # Absolutised, because it is an executable the core launches. An empty pin
            # means "whichever interpreter can browse", so the key stays out of the file.
            self.assertEqual(config["web_python"], str(Path(directory) / "browsenv" / "python"))
            self.assertNotIn("web_python", wizard.build_config({"root": directory}, home=directory))
            self.assertNotIn("web_python",
                             wizard.build_config({"root": directory, "web_python": "   "}, home=directory))
            # There is no second browser agent to choose, so the selector that used to
            # name one is refused by both doors: setup will not accept it as an option,
            # and the writer will not put it in a file the core has to tolerate.
            with self.assertRaisesRegex(ValueError, "Unsupported setup option: web_engine"):
                wizard.build_config({"root": directory, "web_engine": "jev"}, home=directory)
            with self.assertRaisesRegex(ValueError, "Unsupported configuration key: web_engine"):
                wizard.validate_config({"web_engine": "jev"})


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
    def test_an_engine_needs_a_built_native_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            self.assertFalse(wizard.validate_engine(engine)["ok"])
            self.assertFalse(wizard.validate_engine(engine)["is_directory"])
            engine.mkdir()
            self.assertTrue(wizard.validate_engine(engine)["is_directory"])
            self.assertFalse(wizard.validate_engine(engine)["ok"])
            (engine / "Cargo.toml").write_text("", encoding="utf-8")
            self.assertFalse(wizard.validate_engine(engine)["ok"],
                             "a source tree without a built binary is not an engine")
            make_engine(engine)
            report = wizard.validate_engine(engine)
            self.assertTrue(report["ok"])
            self.assertEqual(report["kind"], "fennec")

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

    def test_model_readiness_says_what_the_engine_would_do_with_the_path(self):
        # fennec serve converts nothing and validates nothing before it mmaps a folder, so a
        # not-ready model must be named before the user types the command, not after it
        # answers in fluent nonsense.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = make_model(root, "ready")
            partial = make_model(root, "partial", weights=False)
            models = wizard.scan_models(root)
        self.assertTrue(wizard.model_readiness(ready, models).startswith("ready"),
                        wizard.model_readiness(ready, models))
        line = wizard.model_readiness(partial, models)
        self.assertIn("fennec-convert", line)
        self.assertIn("partial", line)

    def test_model_readiness_never_calls_an_unscanned_path_valid(self):
        # A model outside models_root has no classifier result, so the honest answer is that
        # the path is unchecked rather than a readiness the wizard did not compute.
        self.assertIn("unchecked", wizard.model_readiness("elsewhere/model", []))
        self.assertEqual(wizard.model_readiness("", []), "none selected")


class ModelDownloadTests(unittest.TestCase):
    def test_missing_dependency_gives_an_actionable_install_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
                with self.assertRaises(wizard.ModelDownloadError) as raised:
                    wizard.download_recommended_model(root, environ={}, python=sys.executable)
            message = str(raised.exception)
            self.assertIn("huggingface_hub>=0.25", message)
            self.assertIn("pip install", message)
            self.assertIn("fennec-core[models]", message)
            self.assertIn("--download-model", message)
            self.assertFalse((root / "qwen36-35b").exists())

    def test_success_download_uses_the_verified_repo_safe_files_and_bounded_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"])
                target.mkdir(parents=True)
                for name in wizard.RECOMMENDED_MODEL_FILES:
                    (target / name).write_bytes(b"weights")
                return str(target)

            environ = {"HF_TOKEN": "hf-test-token"}
            path = wizard.download_recommended_model(
                root, environ=environ, snapshot_download=fake_download, out=lambda message: None)

            self.assertEqual(path, root / "qwen36-35b")
            self.assertEqual(len(calls), 1)
            call = calls[0]
            self.assertEqual(call["repo_id"], "Liso/qwen36-35b-colibri")
            self.assertEqual(Path(call["local_dir"]), path)
            self.assertEqual(call["token"], "hf-test-token")
            self.assertEqual(call["max_workers"], 8)
            self.assertLessEqual(call["max_workers"], 8)
            for pattern in ("*.json", "*.safetensors", "*.txt", "tokenizer.model"):
                self.assertIn(pattern, call["allow_patterns"])
            self.assertFalse(any(".gguf" in pattern or ".bin" in pattern
                                 for pattern in call["allow_patterns"]))
            self.assertEqual(environ, {"HF_TOKEN": "hf-test-token"})

    def test_incomplete_download_is_rejected_without_deleting_existing_files(self):
        for present in ("config.json", "model.safetensors"):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                calls = []

                def fake_download(**kwargs):
                    calls.append(kwargs)
                    target = Path(kwargs["local_dir"])
                    target.mkdir(parents=True)
                    (target / present).write_bytes(b"partial")
                    (target / "keep.user").write_text("mine", encoding="utf-8")
                    return str(target)

                with self.assertRaises(wizard.ModelDownloadError) as raised:
                    wizard.download_recommended_model(
                        root, environ={}, snapshot_download=fake_download, out=lambda message: None)
                missing = ".safetensors" if present == "config.json" else "config.json"
                self.assertIn(missing, str(raised.exception))
                self.assertIs(calls[0]["token"], False)
                self.assertEqual((root / "qwen36-35b" / "keep.user").read_text(encoding="utf-8"),
                                 "mine")

    def test_one_missing_or_zero_byte_required_shard_is_incomplete(self):
        for state in ("missing", "empty"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = make_recommended_model(root)
                shard = model / "model-00039.safetensors"
                if state == "missing":
                    shard.unlink()
                else:
                    shard.write_bytes(b"")
                with self.assertRaises(wizard.ModelDownloadError) as raised:
                    wizard.download_recommended_model(
                        root, environ={}, snapshot_download=lambda **_: None,
                        out=lambda message: None)
                self.assertIn("model-00039.safetensors", str(raised.exception))

    def test_complete_model_is_idempotent_and_never_calls_the_downloader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = make_recommended_model(root)
            (model / "keep.user").write_text("mine", encoding="utf-8")
            download = mock.Mock(side_effect=AssertionError("network must not be used"))
            path = wizard.download_recommended_model(
                root, environ={}, snapshot_download=download, out=lambda message: None)
            self.assertEqual(path, model)
            download.assert_not_called()
            self.assertEqual((model / "keep.user").read_text(encoding="utf-8"), "mine")


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

    def run_main(self, *arguments, input_fn=None, environ=None):
        out, err = io.StringIO(), io.StringIO()
        dependencies = {"missing": [], "installed": list(wizard.CORE_DEPS)}
        with mock.patch.object(wizard, "probe_dependencies", return_value=dependencies):
            code = wizard.main(
                ["--root", str(self.root / "Fennec"), "--engine-root", str(self.engine),
                 "--models-root", str(self.models), "--python", sys.executable, *arguments],
                home=self.home, environ={} if environ is None else environ,
                input_fn=input_fn, out=out.write, err=err.write)
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

    def test_the_written_config_advice_states_model_readiness(self):
        code, out, err = self.run_main("--yes")
        self.assertEqual(code, 0, err)
        self.assertIn("readiness ready", out)
        self.assertIn("fennec serve --model", out)

    def test_dry_run_prints_without_writing(self):
        code, out, err = self.run_main("--dry-run", "--json")
        self.assertEqual(code, 0, err)
        printed = json.loads(out)
        self.assertEqual(printed["engine_root"], str(self.engine))
        self.assertFalse(self.config_file.exists())

    def test_setup_never_downloads_without_the_explicit_flag(self):
        with mock.patch.object(wizard, "download_recommended_model",
                               side_effect=AssertionError("unexpected download")) as download:
            code, out, err = self.run_main("--yes")
        self.assertEqual(code, 0, err)
        download.assert_not_called()

    def test_download_model_selects_the_recommended_path_without_persisting_the_token(self):
        make_model(self.models, "other")
        token = "hf-test-secret"
        with mock.patch.object(wizard, "download_recommended_model",
                               return_value=self.model) as download:
            code, out, err = self.run_main(
                "--yes", "--json", "--download-model", environ={"HF_TOKEN": token})
        self.assertEqual(code, 0, err)
        download.assert_called_once_with(
            str(self.models), environ={"HF_TOKEN": token}, python=sys.executable, out=mock.ANY)
        config_text = self.config_file.read_text(encoding="utf-8")
        config = json.loads(config_text)
        self.assertEqual(config["model"], str(self.model))
        self.assertNotIn(token, config_text + out + err)

    def test_incomplete_download_fails_before_writing_the_config(self):
        with mock.patch.object(
                wizard, "download_recommended_model",
                side_effect=wizard.ModelDownloadError("incomplete model download")):
            code, out, err = self.run_main("--yes", "--download-model")
        self.assertEqual(code, 1)
        self.assertIn("incomplete model download", err)
        self.assertFalse(self.config_file.exists())

    def test_dry_run_never_performs_a_requested_download(self):
        with mock.patch.object(wizard, "download_recommended_model") as download:
            code, out, err = self.run_main("--dry-run", "--json", "--download-model")
        self.assertEqual(code, 0, err)
        download.assert_not_called()
        self.assertIn("not performed during --dry-run", err)
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

    def test_the_summary_names_the_browser_decider_and_whether_it_can_run(self):
        # Setup used to print every lane but the one that needs a credential, so a
        # machine with no key reported green and the first web_task failed.
        lines = wizard._summary(
            {"engine_root": "e", "suite_root": "s", "models_root": "m", "cache_root": "c",
             "backend": "vulkan", "ctx": 0, "policy": "ask", "model": ""},
            {"ok": True}, {"missing": []},
            {"os": "Linux", "arch": "x86_64", "cores": 8, "ram_gb": 32, "free_gb": 20},
            {},
        )
        browser = [line for line in lines if line.startswith("browser")]
        self.assertEqual(len(browser), 1, lines)
        self.assertIn("not ready", browser[0])
        self.assertIn("LIQUID_API_KEY", browser[0])
        self.assertNotIn("openbot", "".join(lines))

    def test_the_browser_lane_reads_the_same_env_as_the_decider(self):
        self.assertIn("loopback, free", wizard.browser_lane(
            {"LIQUID_ENDPOINT": "http://127.0.0.1:8199/v1/systemone"}))
        self.assertIn("hosted, billed", wizard.browser_lane({"LIQUID_API_KEY": "k"}))
        # A named endpoint that is not loopback still needs the key: the wizard must
        # not talk itself into "free" because someone typed a URL.
        self.assertIn("not ready", wizard.browser_lane(
            {"LIQUID_ENDPOINT": "https://example.test/v1/systemone"}))


if __name__ == "__main__":
    unittest.main()

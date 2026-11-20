"""The wheel is a list of top-level modules, so the list has to match the checkout.

A module missing from ``[tool.setuptools] py-modules`` installs as nothing at all, and the
console script then dies with ``ModuleNotFoundError`` only on the machines that installed
from the wheel. Reading the directory back is the whole point of this test; the depth
assertions pin the same thing for the two places a path is computed from ``__file__``.
"""
import os
import unittest
from pathlib import Path
from unittest import mock, skipIf

import hermetic  # noqa: F401

try:
    import tomllib  # 3.11+; the project still supports 3.10
except ImportError:  # pragma: no cover - only reached on 3.10
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent


def pyproject():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def modules_on_disk():
    return {path.stem for path in ROOT.glob("*.py")}


@skipIf(tomllib is None, "tomllib needs Python 3.11")
class PackagingTests(unittest.TestCase):
    def test_every_root_module_is_declared(self):
        self.assertEqual(set(pyproject()["tool"]["setuptools"]["py-modules"]), modules_on_disk())

    def test_no_nested_package_reappeared(self):
        nested = sorted(path.parent.name for path in ROOT.glob("*/*.py")
                        if (path.parent / "__init__.py").is_file())
        self.assertEqual(nested, [])

    def test_console_scripts_point_at_declared_modules(self):
        declared = set(pyproject()["tool"]["setuptools"]["py-modules"])
        scripts = pyproject()["project"]["scripts"]
        self.assertEqual(sorted(scripts), ["fennec-agent", "fennec-convert"])
        for name, target in scripts.items():
            self.assertIn(target.split(":")[0], declared, name)

    def test_the_version_lives_in_one_place(self):
        from version import __version__

        self.assertEqual(__version__, pyproject()["project"]["version"])


class DriverPathTests(unittest.TestCase):
    """Every child process reaches this checkout by depth, not by package name."""

    def test_the_web_driver_sits_beside_the_loop_it_imports(self):
        import outlets

        root = Path(outlets.__file__).resolve().parent
        self.assertTrue((root / "web_driver.py").is_file())
        self.assertTrue((root / "webagent.py").is_file())

    def test_the_rust_binary_hint_is_one_directory_above_the_modules(self):
        import rust_bridge

        here = Path(rust_bridge.__file__).resolve().parent
        self.assertEqual(rust_bridge.ROOT, here.parent)
        # FENNEC_RS overrides the default search, so ask for the default.
        with mock.patch.dict(os.environ, {}, clear=True):
            first = Path(next(rust_bridge.candidate_paths()))
        self.assertEqual(first.parent.parent.parent, rust_bridge.ROOT / "fennec-cli")


if __name__ == "__main__":
    unittest.main()

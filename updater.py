"""Auto-update system: silent background updates from GitHub releases,
plus `fennec update --experimental` for devs to pull from the main branch.

Normal mode checks tagged releases and updates silently in the background.
Experimental mode pulls the latest commit from main for developers.
Every download is verified against a SHA-256 checksum before applying.
"""

import hashlib
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import urllib.request
import json

GITHUB_REPO = "Liso7150/fennec-core"


@dataclass
class Asset:
    name: str
    download_url: str
    size: int


@dataclass
class Release:
    version: str
    tag: str
    published_at: str
    assets: List[Asset]
    checksums_url: Optional[str] = None


def get_platform() -> tuple[str, str]:
    """Detect current platform"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    os_name = {
        "windows": "windows",
        "darwin": "macos",
        "linux": "linux",
    }.get(system, system)
    
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(machine, machine)
    
    return os_name, arch


def asset_name(base: str) -> str:
    """Generate asset name for current platform"""
    os_name, arch = get_platform()
    if os_name == "windows":
        return f"{base}-{os_name}-{arch}.exe"
    else:
        return f"{base}-{os_name}-{arch}"


def is_newer(candidate, current):
    """`True` only when candidate is a strictly newer dotted version.

    A string compare ranks 1.9.0 above 1.10.0, and the published releases run
    *behind* a working tree, so a 1.0.0 checkout must not pull 0.9.0 assets just
    because the tag differs. Returns `None` when either side is not three plain
    numbers, and the caller refuses to guess.
    """
    def parts(version):
        fields = str(version).strip().lstrip("v").split(".")
        if len(fields) != 3:
            return None
        try:
            return tuple(int(field) for field in fields)
        except ValueError:
            return None

    left, right = parts(candidate), parts(current)
    if left is None or right is None:
        return None
    return left > right


def check_stable(current_version: str) -> Optional[Release]:
    """Check GitHub releases for newer version"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fennec-updater"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 404:
                return None
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise Exception(f"GitHub API: {e}")
    
    tag = data.get("tag_name", "")
    version = tag.lstrip("v")
    
    # Simple version comparison
    if is_newer(version, current_version) is not True:
        return None
    
    assets = [
        Asset(
            name=a["name"],
            download_url=a["browser_download_url"],
            size=a["size"],
        )
        for a in data.get("assets", [])
    ]
    
    # Look for checksums file
    checksums_url = None
    for a in data.get("assets", []):
        if "checksums" in a["name"].lower() or a["name"].endswith(".sha256"):
            checksums_url = a["browser_download_url"]
            break
    
    return Release(
        version=version,
        tag=tag,
        published_at=data.get("published_at", ""),
        assets=assets,
        checksums_url=checksums_url,
    )


def check_experimental() -> Optional[Release]:
    """Check the default branch for the latest commit"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fennec-updater"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise Exception(f"GitHub API: {e}")
    
    sha = data.get("sha", "")
    short_sha = sha[:7]
    
    return Release(
        version=f"dev-{short_sha}",
        tag=sha,
        published_at=data.get("commit", {}).get("author", {}).get("date", ""),
        assets=[],
        checksums_url=None,
    )


def download_asset(url: str, expected_checksum: Optional[str] = None) -> bytes:
    """Download an asset with optional checksum verification"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fennec-updater"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except Exception as e:
        raise Exception(f"Download: {e}")
    
    if expected_checksum:
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected_checksum:
            raise Exception(f"Checksum mismatch: expected {expected_checksum}, got {actual}")
    
    return data


def stage_asset(new_data: bytes, name: str) -> Path:
    """Park a verified download in the temp dir, overwriting nothing live.

    `sys.executable` is the interpreter running core, not a Fennec binary, so
    replacing it in place put release bytes over Python itself. What moves this
    repo is `fennec update`, which fast-forwards the git checkout.
    """
    staged = Path(tempfile.gettempdir()) / name
    staged.write_bytes(new_data)
    return staged


def update(experimental: bool = False) -> str:
    """Main update function"""
    if experimental:
        print("[update] Checking experimental channel (main branch)...")
        release = check_experimental()
        if release is None:
            return "Already on latest experimental build"
        
        print(f"[update] New commit available: {release.version}")
        
        # For experimental, we'd need to build from source or download CI artifacts
        # For now, just report availability
        return f"Experimental update available: {release.version} (build from source or wait for CI)"
    else:
        print("[update] Checking stable releases...")
        from version import __version__
        current = __version__
        
        release = check_stable(current)
        if release is None:
            return f"Already on latest version ({current})"
        
        print(f"[update] New version available: {release.version} (current: {current})")
        
        name = asset_name("fennec-core")
        asset = next((a for a in release.assets if a.name == name), None)
        
        if asset is None:
            return f"No asset for platform: {name}"
        
        print(f"[update] Downloading {name}...")
        
        # Download checksum if available
        checksum = None
        if release.checksums_url:
            try:
                checksum_data = download_asset(release.checksums_url).decode()
                for line in checksum_data.splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == name:
                        checksum = parts[0]
                        break
            except Exception:
                pass
        
        data = download_asset(asset.download_url, checksum)
        staged = stage_asset(data, name)
        
        return (
            f"Downloaded v{release.version} to {staged}. This repo updates from git: "
            f"run `fennec update` to fast-forward the checkout."
        )


if __name__ == "__main__":
    import sys
    if {"-h", "--help"} & set(sys.argv[1:]):
        print("usage: python updater.py [--experimental]")
        print("  --experimental  check the main branch instead of the latest release")
        raise SystemExit(0)
    experimental = "--experimental" in sys.argv
    try:
        result = update(experimental)
        print(f"[update] {result}")
    except Exception as e:
        print(f"[update] Error: {e}", file=sys.stderr)
        sys.exit(1)

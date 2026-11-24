import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import updater  # noqa: E402


NAME = "fennec-core-windows-x86_64.exe"
SUMS = "https://example.test/SHA256SUMS"
BINARY = "https://example.test/binary"
DIGEST = hashlib.sha256(b"hello").hexdigest()


def response(body):
    result = mock.MagicMock()
    result.status = 200
    result.__enter__.return_value = result
    result.read.return_value = body
    return result


def release(checksums_url=SUMS):
    return updater.Release(
        version="1.0.1",
        tag="v1.0.1",
        published_at="2026-11-24T00:00:00Z",
        assets=[updater.Asset(NAME, BINARY, 5)],
        checksums_url=checksums_url,
    )


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        platform = mock.patch.object(updater, "get_platform", return_value=("windows", "x86_64"))
        platform.start()
        self.addCleanup(platform.stop)

    def test_manifest_requires_exact_basename_and_normalizes_case(self):
        text = f"{'a' * 64}  other.exe\n{DIGEST.upper()} *{NAME}\n"
        self.assertEqual(updater.checksum_for(text, NAME), DIGEST)
        for filename in (f"./{NAME}", f"dir/{NAME}", f"{NAME}.bak", f"prefix-{NAME}"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "no entry"):
                    updater.checksum_for(f"{DIGEST}  {filename}\n", NAME)

    def test_manifest_rejects_malformed_and_duplicate_entries(self):
        for text in (
            f"{'a' * 63}  {NAME}\n",
            f"{'a' * 65}  {NAME}\n",
            f"{'g' * 64}  {NAME}\n",
            "missing-fields\n",
            f"{'a' * 64}  other.exe extra\n",
            f"{DIGEST}  {NAME}\n{DIGEST} *{NAME}\n",
        ):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "SHA256SUMS"):
                updater.checksum_for(text, NAME)

    def test_download_validates_digest_and_compares_lowercase(self):
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=response(b"hello")) as opened:
            self.assertEqual(updater.download_asset(BINARY, DIGEST.upper()), b"hello")
            opened.assert_called_once()
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=response(b"hello")):
            with self.assertRaisesRegex(Exception, "Checksum mismatch"):
                updater.download_asset(BINARY, "0" * 64)
        with mock.patch.object(updater.urllib.request, "urlopen") as opened:
            for digest in ("", "a" * 63, "a" * 65, "g" * 64):
                with self.subTest(digest=digest), self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                    updater.download_asset(BINARY, digest)
            opened.assert_not_called()

    def test_stable_check_selects_only_the_exact_checksums_asset(self):
        payload = {
            "tag_name": "v1.0.1",
            "published_at": "2026-11-24T00:00:00Z",
            "assets": [
                {"name": "checksums.txt", "browser_download_url": "https://example.test/wrong", "size": 1},
                {"name": "SHA256SUMS", "browser_download_url": SUMS, "size": 1},
            ],
        }
        import json
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=response(json.dumps(payload).encode())):
            self.assertEqual(updater.check_stable("1.0.0").checksums_url, SUMS)
        payload["assets"].pop()
        with mock.patch.object(updater.urllib.request, "urlopen", return_value=response(json.dumps(payload).encode())):
            self.assertIsNone(updater.check_stable("1.0.0").checksums_url)

    def assert_update_rejected_without_staging(self, selected_release, bodies, message, expected_urls):
        requested = []

        def open_url(request, timeout):
            requested.append(request.full_url)
            return response(bodies[request.full_url])

        with mock.patch.object(updater, "check_stable", return_value=selected_release), \
                mock.patch.object(updater.urllib.request, "urlopen", side_effect=open_url), \
                mock.patch.object(updater, "stage_asset") as stage:
            with self.assertRaisesRegex(Exception, message):
                updater.update()
        stage.assert_not_called()
        self.assertEqual(requested, expected_urls)

    def test_update_requires_the_checksums_asset_before_any_download(self):
        self.assert_update_rejected_without_staging(release(None), {}, "missing SHA256SUMS", [])

    def test_update_rejects_missing_malformed_and_non_utf8_manifests_without_staging(self):
        cases = (
            (f"{'a' * 64}  other.exe\n".encode(), "no entry"),
            (f"short  {NAME}\n".encode(), "64 hexadecimal"),
            (b"\xff\xfe", "SHA256SUMS"),
        )
        for body, message in cases:
            with self.subTest(body=body):
                self.assert_update_rejected_without_staging(
                    release(), {SUMS: body}, message, [SUMS]
                )

    def test_update_rejects_checksum_fetch_failure_without_staging(self):
        with mock.patch.object(updater, "check_stable", return_value=release()), \
                mock.patch.object(updater.urllib.request, "urlopen", side_effect=OSError("offline")) as opened, \
                mock.patch.object(updater, "stage_asset") as stage:
            with self.assertRaisesRegex(RuntimeError, "Download SHA256SUMS"):
                updater.update()
        opened.assert_called_once()
        stage.assert_not_called()

    def test_update_rejects_a_mismatched_binary_without_staging(self):
        bodies = {
            SUMS: f"{'0' * 64}  {NAME}\n".encode(),
            BINARY: b"hello",
        }
        self.assert_update_rejected_without_staging(
            release(), bodies, "Checksum mismatch", [SUMS, BINARY]
        )

    def test_update_stages_only_a_verified_binary(self):
        bodies = {
            SUMS: f"{DIGEST.upper()} *{NAME}\n".encode(),
            BINARY: b"hello",
        }
        requested = []

        def open_url(request, timeout):
            requested.append(request.full_url)
            return response(bodies[request.full_url])

        with mock.patch.object(updater, "check_stable", return_value=release()), \
                mock.patch.object(updater.urllib.request, "urlopen", side_effect=open_url), \
                mock.patch.object(updater, "stage_asset", return_value=Path("staged")) as stage:
            result = updater.update()
        self.assertIn("Downloaded v1.0.1", result)
        self.assertEqual(requested, [SUMS, BINARY])
        stage.assert_called_once_with(b"hello", NAME)


if __name__ == "__main__":
    unittest.main()

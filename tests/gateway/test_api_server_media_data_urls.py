"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.  Images larger than a
display-size budget are re-encoded instead of inlined raw, because every
downstream hop (frontend SSE line reader, socket payloads, DB rows,
request-body cap) has a hard size ceiling — multi-MB base64 in the
message text surfaced as 400/413 errors and silently dropped SSE events.
"""

import base64
import io
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms import api_server as mod
from gateway.platforms.api_server import _resolve_media_to_data_urls  # noqa: E402

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


def _noise_png(edge: int = 1400) -> bytes:
    """A large incompressible RGB PNG (>256 KiB) for re-encode tests."""
    import os

    from PIL import Image

    img = Image.frombytes("RGB", (edge, edge), os.urandom(edge * edge * 3))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_bytes(self, data: bytes, name: str = "shot.png", prefix: str = "hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=prefix))
        p = d / name
        p.write_bytes(data)
        return p

    def _write_png(self, tmpdir_name="hermes_media_test"):
        return self._write_bytes(_PNG_BYTES, prefix=tmpdir_name)

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_text_without_media_passthrough(self):
        self.assertEqual(_resolve_media_to_data_urls("plain text"), "plain text")
        self.assertEqual(_resolve_media_to_data_urls(""), "")

    def test_small_image_inlined_byte_for_byte(self):
        # The 1x1 PNG is under the original-inline threshold — its exact
        # bytes (and thus exact data URL) must be preserved.
        p = self._write_png()
        expected = b"data:image/png;base64," + base64.b64encode(_PNG_BYTES)
        self.assertIn(expected.decode(), _resolve_media_to_data_urls(f"MEDIA:{p}"))

    def test_large_image_reencoded_bounded(self):
        """Oversized images are re-encoded to a display-size JPEG instead of
        inlining megabytes of raw base64 — this is what keeps every
        downstream consumer (aiohttp line reader, socket, request cap)
        inside its limits for real image turns."""
        from PIL import Image

        p = self._write_bytes(_noise_png())
        raw = p.read_bytes()
        raw_b64_len = (len(raw) * 4) // 3
        out = _resolve_media_to_data_urls(f"MEDIA:{p}")
        self.assertIn("data:image/jpeg;base64,", out)
        data_url = out.split("data:image/jpeg;base64,")[1].split(")")[0]
        self.assertLess(len(data_url), raw_b64_len)
        decoded = base64.b64decode(data_url)
        img = Image.open(io.BytesIO(decoded))
        self.assertLessEqual(max(img.size), mod._MEDIA_INLINE_MAX_EDGE)

    def test_undecodable_image_left_untouched(self):
        """A path with an image suffix but undecodable bytes must not be
        inlined (would emit a broken image) — the MEDIA: tag stays literal."""
        # Oversized so the re-encode path runs (small files pass through
        # byte-for-byte regardless of content).
        p = self._write_bytes(b"not an image " * (24 * 1024))
        text = f"MEDIA:{p}"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_per_message_budget_exhausted_leaves_tag(self):
        orig = mod._MEDIA_INLINE_TOTAL_BUDGET_BYTES
        mod._MEDIA_INLINE_TOTAL_BUDGET_BYTES = 1
        try:
            p = self._write_png()
            self.assertEqual(_resolve_media_to_data_urls(f"MEDIA:{p}"), f"MEDIA:{p}")
        finally:
            mod._MEDIA_INLINE_TOTAL_BUDGET_BYTES = orig

    def test_multiple_tags(self):
        p1 = self._write_png()
        p2 = self._write_png("hermes_media_test2")
        out = _resolve_media_to_data_urls(f"MEDIA:{p1}\nand MEDIA:{p2}")
        self.assertEqual(out.count("data:image/png;base64,"), 2)

    def test_relative_traversal_path_not_inlined(self):
        """A relative/traversal path must never be inlined — the anchored
        MEDIA_TAG_CLEANUP_RE matcher requires an absolute-path prefix
        (~/, /, or a Windows drive letter), so a bare relative token after
        MEDIA: is left as literal text rather than resolved against cwd."""
        text = "MEDIA:../../../../etc/passwd.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_credential_path_not_inlined_even_with_image_extension(self):
        """An absolute path under the credential/system-path denylist
        (validate_media_delivery_path) must not be inlined even though it
        has an allowed image extension and the tag matcher's shape."""
        text = "MEDIA:~/.ssh/id_rsa.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_symlink_escaping_to_denylisted_target_not_inlined(self):
        """A symlink whose resolved target lands under a denylisted system
        prefix (/etc) must not be inlined — validate_media_delivery_path
        resolves symlinks before the containment/denylist check runs, so
        the traversal can't be laundered through an innocuous-looking
        image-suffixed symlink name."""
        import os
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix="hermes_media_test_symlink"))
        link = d / "shot.png"
        try:
            os.symlink("/etc/hosts", link)
        except OSError:
            self.skipTest("symlink creation not supported in this environment")
        text = f"MEDIA:{link}"
        self.assertEqual(_resolve_media_to_data_urls(text), text)


class TestCaptureEmbeddedDataUrls(unittest.TestCase):
    """Model-embedded ``data:image`` blobs are captured to media-cache files
    and rewritten as MEDIA: tags so the same bounded inline pipeline applies.
    Without this, a turn where the model streams raw base64 into its reply
    blows the same size limits the MEDIA: inlining exists to avoid."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._orig_dir = mod.IMAGE_CACHE_DIR
        self._cache = Path(tempfile.mkdtemp(prefix="hermes_media_cache_test"))
        mod.IMAGE_CACHE_DIR = self._cache

    def tearDown(self):
        mod.IMAGE_CACHE_DIR = self._orig_dir

    def _blob(self, edge: int = 1400) -> bytes:
        return _noise_png(edge)

    def test_large_embedded_blob_become_media_tag_then_bounded_data_url(self):
        from gateway.platforms.api_server import _capture_embedded_data_urls, _prepare_display_text

        blob = self._blob()
        big_b64 = base64.b64encode(blob).decode()
        text = f"Here it is: ![Bedroom POV 1](data:image/png;base64,{big_b64})"
        rewritten = _capture_embedded_data_urls(text)
        self.assertIn("MEDIA:", rewritten)
        self.assertNotIn("data:image", rewritten)
        # The blob landed in the media cache with an image extension.
        cached = [p for p in self._cache.iterdir() if p.suffix in (".png", ".jpg")]
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0].read_bytes(), blob)
        # Full display pipeline re-inlines it bounded.
        out = _prepare_display_text(text)
        self.assertIn("data:image/", out)
        self.assertNotIn("MEDIA:", out)
        data_url = out.split("data:image/")[1]
        self.assertLess(len(data_url), len(big_b64))

    def test_small_embedded_blob_left_inline(self):
        from gateway.platforms.api_server import _capture_embedded_data_urls

        small_b64 = base64.b64encode(_PNG_BYTES).decode()
        text = f"![tiny](data:image/png;base64,{small_b64})"
        self.assertEqual(_capture_embedded_data_urls(text), text)
        self.assertEqual(list(self._cache.iterdir()), [])

    def test_multiple_blobs_captured_with_budget_limit(self):
        from gateway.platforms.api_server import _capture_embedded_data_urls

        mod._MEDIA_CAPTURE_MAX_COUNT = 2
        try:
            text = " ".join(
                f"![{i}](data:image/png;base64,{base64.b64encode(self._blob()).decode()})"
                for i in range(3)
            )
            out = _capture_embedded_data_urls(text)
            self.assertEqual(out.count("MEDIA:"), 2)
            self.assertEqual(len(list(self._cache.iterdir())), 2)
        finally:
            mod._MEDIA_CAPTURE_MAX_COUNT = 20


class TestBoundedNormalizedText(unittest.TestCase):
    """Inline base64 image data is stripped from model-context text parts —
    it is unreadable pixel noise to the LLM and only inflates the prompt —
    before the standard 64 KB cap applies."""

    def test_strips_data_url_and_truncates(self):
        from gateway.platforms.api_server import _bounded_normalized_text

        big_b64 = base64.b64encode(_noise_png(200)).decode()
        text = f"previous message with ![img](data:image/png;base64,{big_b64}) end"
        out = _bounded_normalized_text(text)
        self.assertNotIn("data:image", out)
        self.assertIn("[base64 image]", out)
        # Truncation still applies to oversized text.
        huge = "x" * 100_000
        self.assertLessEqual(len(_bounded_normalized_text(huge)), mod.MAX_NORMALIZED_TEXT_LENGTH)


if __name__ == "__main__":
    unittest.main()

"""Round 6n, defect 5: artifact URLs in the console encode each path segment.

artifactUrl in web/src/lib/api.ts used to put the file path into the URL as it was, so "chart #1.png" requested "chart "
and a "%" in a name was mis-decoded. It now encodes each segment with encodeURIComponent, the way tableDownloadUrl does,
and keeps the "/" between segments.

The test loads the real api.ts with Node's built-in type stripping, so no packages are installed. It skips when Node is
not available.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

API_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "api.ts"
LOADER = ("const [url, fn, ...args] = process.argv.slice(1);"
          "const mod = await import(url);"
          "const out = fn === 'artifact' ? mod.artifactUrl(...args) : mod.tableDownloadUrl(...args);"
          "console.log(JSON.stringify(out));")


def _call(fn: str, *args) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the web module cannot be loaded")
    proc = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings", "--input-type=module", "-e", LOADER,
         "--", API_TS.as_uri(), fn, *(str(a) for a in args)],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("path, expected", [
    ("chart #1.png", "/api/artifacts/run1/chart%20%231.png"),
    ("out/chart #1.png", "/api/artifacts/run1/chart%20%231.png"),          # the out/ prefix is dropped, then encoded
    ("100%.csv", "/api/artifacts/run1/100%25.csv"),                          # a percent sign is encoded, not mis-decoded
    ("sub dir/a?b.png", "/api/artifacts/run1/sub%20dir/a%3Fb.png"),          # "/" between segments is kept
    ("out/plain.json", "/api/artifacts/run1/plain.json"),                    # plain names are unchanged
])
def test_artifact_paths_are_encoded_one_segment_at_a_time(path, expected):
    assert _call("artifact", "run1", path) == expected


def test_the_run_id_is_encoded_as_one_segment():
    assert _call("artifact", "run/1", "out/a.csv") == "/api/artifacts/run%2F1/a.csv"


def test_table_download_urls_keep_their_encoding():
    assert _call("table", "nb1", "sales #2", 3) == "/api/notebooks/nb1/tables/sales%20%232/download?version=3"

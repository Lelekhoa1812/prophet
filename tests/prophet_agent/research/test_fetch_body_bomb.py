"""A gzip body is decoded in capped pieces, so a decompression bomb cannot fill memory before the byte cap is applied
(GOAL.md Round 6m, defect b). The test serves 64 MiB of text as about 65 KB of gzip from a real local HTTP server."""
import gzip
import threading
import tracemalloc
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from prophet_agent.research import netguard
from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle

MIB = 1024 * 1024
PUBLIC = "93.184.216.34"
BOMB = gzip.compress(b"a" * (64 * MIB), compresslevel=9)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/robots.txt":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(BOMB)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(BOMB)

    def log_message(self, *args):
        pass


class _RouteToLocal(httpx.BaseTransport):
    """Sends each request to the local test server. The fetcher has already checked and pinned the address in the URL;
    only the socket is moved, and the Host header keeps the name."""

    def __init__(self, port):
        self._inner = httpx.HTTPTransport()
        self._port = port

    def handle_request(self, request):
        request.url = request.url.copy_with(host="127.0.0.1", port=self._port)
        return self._inner.handle_request(request)


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_gzip_bomb_is_capped_while_it_is_decoded(tmp_path, monkeypatch, local_server):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    client = httpx.Client(transport=_RouteToLocal(local_server), follow_redirects=False)
    tracemalloc.start()
    try:
        page = fetch_page("http://news.example.com/bomb", client=client, resolver=lambda host, port: [PUBLIC],
                          throttle=HostThrottle(0.0), use_cache=False, max_bytes=2 * MIB)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        client.close()
    assert page.ok is True
    assert "response cut at" in page.note
    assert peak < 16 * MIB, f"peak {peak / MIB:.1f} MiB: the decoded body was held whole before the cap was applied"


class _Pieces(httpx.SyncByteStream):
    """A stream that is not read eagerly, as a network response is: the fetcher reads the raw pieces itself."""

    def __init__(self, pieces):
        self._pieces = pieces

    def __iter__(self):
        yield from self._pieces


def _streamed(content_encoding, pieces):
    headers = {"content-encoding": content_encoding} if content_encoding else {}
    return httpx.Response(200, headers=headers, stream=_Pieces(pieces))


@pytest.mark.parametrize("encoding", ["gzip", "x-gzip"])
def test_streamed_gzip_decodes_to_the_original_text(encoding):
    text = b"Brent crude rose again. " * 500
    raw = gzip.compress(text)
    pieces = [raw[i:i + 7] for i in range(0, len(raw), 7)]   # small pieces, so decoding spans many reads
    assert netguard.read_capped(_streamed(encoding, pieces), 10 * MIB) == (text, "")


def test_streamed_raw_deflate_without_a_zlib_header_is_still_read():
    text = b"Copper eased on Tuesday. " * 200
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = compressor.compress(text) + compressor.flush()
    assert netguard.read_capped(_streamed("deflate", [raw]), 10 * MIB) == (text, "")


def test_a_coding_this_reader_does_not_decode_is_refused_before_the_body_is_read():
    assert netguard.read_capped(_streamed("br", [b"\x0b\x00"]), 1024) == (b"", "encoding")


def test_a_damaged_gzip_body_is_refused():
    assert netguard.read_capped(_streamed("gzip", [b"not gzip at all"]), 1024) == (b"", "encoding")

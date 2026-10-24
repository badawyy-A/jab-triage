"""The real urllib transport, against a throwaway server on localhost. No call
leaves the machine."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jab_triage.backends.base import BackendError, FatalBackendError
from jab_triage.backends.systemone import Endpoint, SystemOneBackend
from jab_triage.presets import Question

QUESTIONS = (Question("dup", "noul", "Duplicate?"),)
OK = {
    "model": "local",
    "answers": {"dup": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 5, "output_tokens": 1},
}


@pytest.fixture
def server():
    """Replies with the queued (status, headers, body) tuples, in order."""
    replies = []
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append({"auth": self.headers["Authorization"], "body": self.rfile.read(length)})
            status, headers, body = replies.pop(0)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/v1/systemone"
    yield url, replies, seen
    httpd.shutdown()
    httpd.server_close()


def make_backend(url, slept):
    return SystemOneBackend("local", Endpoint(url, "LOCAL_KEY"), "k-123", sleep=slept.append)


def test_round_trip(server):
    url, replies, seen = server
    replies.append((200, {}, json.dumps(OK).encode()))
    decision = make_backend(url, []).decide("some state", QUESTIONS)
    assert decision.answers["dup"].value == 0.9
    assert seen[0]["auth"] == "Bearer k-123"
    assert json.loads(seen[0]["body"])["state"] == "some state"


def test_error_status_and_retry_after_come_through(server):
    url, replies, _ = server
    replies.append((429, {"Retry-After": "2"}, b"{}"))
    replies.append((200, {}, json.dumps(OK).encode()))
    slept = []
    make_backend(url, slept).decide("s", QUESTIONS)
    assert len(slept) == 1
    assert 1.9 < slept[0] <= 2.0


def test_rejected_key_is_fatal(server):
    url, replies, _ = server
    replies.append((401, {}, b'{"error": {"code": 401}}'))
    with pytest.raises(FatalBackendError, match="LOCAL_KEY"):
        make_backend(url, []).decide("s", QUESTIONS)


def test_connection_refused_is_a_network_error():
    # Port 9 (discard) is almost never open on localhost.
    backend = make_backend("http://127.0.0.1:9/v1/systemone", [])
    with pytest.raises(BackendError, match="network error"):
        backend.decide("s", QUESTIONS)

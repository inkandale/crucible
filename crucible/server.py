"""POST /score: run the crucible on the card in the request, answering in the shape the enrich plugin stores."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 16 << 20  # the plugin sends the whole card, lorebook included


class Server(ThreadingHTTPServer):
    # The plugin sends a batch's cards at once and Innkeeper runs several batches; the default backlog of 5 resets the rest.
    request_queue_size = 1024
    daemon_threads = True


def make_server(crucible, host: str, port: int) -> Server:
    meta = {"source": "crucible", "version": crucible.cfg["version"]}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: dict):
            out = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self.reply(200, {"ok": True}) if self.path == "/healthz" else self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self.reply(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    return self.reply(413, {"error": f"cards over {MAX_BODY} bytes aren't scored"})
                data = json.loads(self.rfile.read(n))["data"]
                if not isinstance(data, dict):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self.reply(400, {"error": "send JSON with the card's data"})
            self.reply(200, {"values": crucible.score(data), "meta": meta})

        def log_message(self, *args):
            pass

    return Server((host, port), Handler)

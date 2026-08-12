"""
la_worker.py — Out-of-process worker for LocateAnything-3B.

Runs inside the ``mtsd-la`` conda env (transformers ~4.57) and serves
predictions over localhost HTTP. The main PromptDetect app runs on transformers
5.x (for SAM 3/3.1 + Cosmos); since LocateAnything's remote code is bound to
transformers 4.57, it can't share that process — so the app launches this worker
transparently and proxies inference to it. **Not meant to be run by hand**; it is
spawned by ``LocateAnythingEngine`` in backend.py.

Protocol (localhost only, JSON):
    GET  /health   -> {"ready": bool, "error": str|null}
    POST /predict  {"image_path","prompt","conf","max_detections"}
                   -> {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}
    POST /shutdown -> {"ok": true}; stops the server
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Force the shared LocateAnythingEngine to load *in-process* here (rather than
# recursively trying to spawn another worker).
os.environ["PROMPTDETECT_LA_WORKER"] = "1"

# Progress/detail strings contain unicode (e.g. "✓"); the default Windows console
# codepage (cp1252) can't encode them and would crash on print. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend import LocateAnythingEngine  # noqa: E402

STATE = {"engine": None, "ready": False, "error": None}


def _load(repo_id: str, device: str) -> None:
    """Load the model once, in a background thread, so /health stays responsive."""
    try:
        eng = LocateAnythingEngine(repo_id, device)
        eng.load(lambda p, m: print(f"[la_worker] {int(p * 100):3d}% {m}", flush=True))
        STATE["engine"] = eng
        STATE["ready"] = True
        print("[la_worker] READY", flush=True)
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print("[la_worker] LOAD FAILED:\n" + traceback.format_exc(), flush=True)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # silence default request logging
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ready": STATE["ready"], "error": STATE["error"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"

        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        if not STATE["ready"]:
            self._send(503, {"error": STATE["error"] or "model still loading"})
            return

        try:
            req = json.loads(raw)
            image = np.array(Image.open(req["image_path"]).convert("RGB"))
            boxes, scores, labels, _masks = STATE["engine"].predict_raw(
                image, req["prompt"], float(req.get("conf", 0.3)),
                int(req.get("max_detections", 100)),
            )
            self._send(200, {"boxes": boxes, "scores": scores, "labels": labels})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--repo", default="nvidia/LocateAnything-3B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    threading.Thread(target=_load, args=(args.repo, args.device), daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[la_worker] serving on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

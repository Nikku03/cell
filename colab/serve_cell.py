"""Serve the interactive cell on localhost.

    python colab/serve_cell.py            # -> http://localhost:8000/cell
    python colab/serve_cell.py 8080       # pick a port

The cell HTML is fully self-contained (data embedded), so you can also just open
outputs/orphan/cell_complete.html directly. This server is here so you get a clean
localhost URL, and so it works from Colab (it prints an ngrok-free local link).
"""
import sys, http.server, socketserver, webbrowser, functools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = ROOT / "outputs" / "orphan" / "cell_complete.html"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

if not CELL.exists():
    sys.exit("cell_complete.html not found — run: python colab/build_cell_complete.py "
             "&& python colab/build_cell_app_complete.py")


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/cell", "/index.html"):
            body = CELL.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *a):  # quiet
        pass


with socketserver.TCPServer(("", PORT), H) as httpd:
    url = f"http://localhost:{PORT}/cell"
    print(f"\n  The cell is live at  {url}\n  (Ctrl-C to stop)\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()

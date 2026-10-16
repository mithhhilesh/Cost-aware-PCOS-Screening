"""Local-only model service and built React app. No notebook kernel or Node needed at runtime."""
import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from model_service import ArtifactStore

STATIC = Path(__file__).resolve().parents[1] / 'dist' / 'public'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *_):
        pass  # No patient request bodies or prediction values in logs.

    def send_json(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/schema':
            return self.send_json(200, self.server.store.schema())
        if path == '/api/comparisons':
            return self.send_json(200, self.server.store.comparisons())
        if path.startswith('/api/'):
            return self.send_json(404, {'error': 'Unknown API endpoint.'})
        if not STATIC.is_dir():
            return self.send_json(503, {'error': 'Frontend build missing. Run npm install and npm run build.'})
        return super().do_GET()

    def do_POST(self):
        origin = self.headers.get('Origin')
        allowed = {f'http://{host}:{port}' for host in ['127.0.0.1', 'localhost']
                   for port in [self.server.server_port, 5173]}
        if origin and origin not in allowed:
            return self.send_json(403, {'error': 'This service accepts the local dashboard only.'})
        path = urlsplit(self.path).path
        if path not in ['/api/predict', '/api/explain']:
            return self.send_json(404, {'error': 'Unknown API endpoint.'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send_json(415, {'error': 'Use application/json.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                return self.send_json(413, {'error': 'Invalid or oversized screening request.'})
            payload = json.loads(self.rfile.read(length))
            result = (self.server.store.predict(payload) if path == '/api/predict' else
                      self.server.store.explain(payload))
            return self.send_json(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            return self.send_json(422, {'error': str(exc)})
        except Exception:
            return self.send_json(500, {'error': 'Model computation failed. Check export/package compatibility; '
                                      'if only the explanation failed, your screening score is still available.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        store = ArtifactStore(args.artifacts)
    except Exception as exc:
        parser.exit(1, f'Cannot open notebook export: {exc}\n')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.store = store
    print(f'PCOS Insight: http://127.0.0.1:{args.port}', flush=True)
    print(f'Using export: {store.directory}', flush=True)
    print('Press Ctrl+C to stop. Restart after exporting new models.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

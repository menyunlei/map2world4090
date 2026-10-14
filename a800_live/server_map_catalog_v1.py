#!/usr/bin/env python3
"""CPU-only, read-only catalog for the user's approved server map folder."""
import argparse
import hashlib
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps

MAPS_ROOT = Path(os.environ.get('DREAMX_MAPS_ROOT', 'inputs/map_build/map'))
ORIGINS = {'http://127.0.0.1:18765', 'http://localhost:18765'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
Image.MAX_IMAGE_PIXELS = 8_000_000


def inventory(root):
    entries = {}
    for candidate in root.iterdir():
        if candidate.suffix.lower() not in EXTENSIONS:
            continue
        try:
            path = candidate.resolve(strict=True)
            if path.parent != root or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size > 12 * 1024 * 1024:
                continue
            with Image.open(path) as image:
                width, height = image.size
                if (image.format not in {'JPEG', 'PNG', 'WEBP'} or
                        getattr(image, 'n_frames', 1) != 1 or
                        min(width, height) < 32 or max(width, height) > 4096 or
                        width * height > 8_000_000):
                    continue
                mime = Image.MIME[image.format]
            identity = hashlib.sha256(candidate.name.encode()).hexdigest()[:24]
            entries[identity] = {'id': identity, 'name': candidate.name,
                                 'width': width, 'height': height,
                                 '_path': path, '_mime': mime}
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    return entries


def natural_key(entry):
    return [int(piece) if piece.isdigit() else piece.lower()
            for piece in re.split(r'(\d+)', entry['name'])]


class CatalogHandler(BaseHTTPRequestHandler):
    server_version = 'DreamXMapCatalog/1'

    def respond(self, status, body, mime='application/json; charset=utf-8'):
        self.send_response(status)
        origin = self.headers.get('Origin')
        if origin in ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Vary', 'Origin')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get('Origin') not in ORIGINS | {None}:
            self.respond(403, b'{"error":"origin not allowed"}')
            return
        request = urlsplit(self.path)
        if request.path == '/health':
            self.respond(200, b'{"ready":true,"cpu_only":true,"read_only":true}')
            return
        if request.path not in {'/maps', '/maps/image', '/maps/thumbnail'}:
            self.respond(404, b'{"error":"not found"}')
            return
        entries = inventory(self.server.maps_root)
        if request.path == '/maps':
            public = [{k: v for k, v in entry.items() if not k.startswith('_')}
                      for entry in sorted(entries.values(), key=natural_key)]
            self.respond(200, json.dumps({'maps': public}, ensure_ascii=False).encode())
            return
        identity = parse_qs(request.query).get('id', [''])[0]
        entry = entries.get(identity)
        if entry is None:
            self.respond(404, b'{"error":"unknown map id"}')
            return
        try:
            if request.path == '/maps/image':
                self.respond(200, entry['_path'].read_bytes(), entry['_mime'])
            else:
                with Image.open(entry['_path']) as image:
                    image = ImageOps.exif_transpose(image)
                    image.thumbnail((220, 260))
                    rgba = image.convert('RGBA')
                    opaque = Image.new('RGB', rgba.size, 'white')
                    opaque.paste(rgba, mask=rgba.getchannel('A'))
                    output = io.BytesIO()
                    opaque.save(output, format='JPEG', quality=85)
                self.respond(200, output.getvalue(), 'image/jpeg')
        except OSError:
            self.respond(404, b'{"error":"map unavailable; refresh catalog"}')


def make_server(port=18769, root=MAPS_ROOT):
    server = ThreadingHTTPServer(('127.0.0.1', port), CatalogHandler)
    server.maps_root = root.resolve(strict=True)
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18769)
    args = parser.parse_args()
    make_server(args.port).serve_forever()

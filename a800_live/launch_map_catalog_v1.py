#!/usr/bin/env python3
"""Single CPU catalog owner; does not import or signal any GPU service."""
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from server_map_catalog_v1 import MAPS_ROOT, make_server

ROOT = Path(__file__).resolve().parent
audit = ROOT / 'audit'
audit.mkdir(exist_ok=True)
guard = open(audit / 'MAP_CATALOG_OWNER_V1.lock', 'a+')
fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
server = make_server()
record = {'pid': os.getpid(), 'started_epoch': time.time(),
          'address': '127.0.0.1:18769', 'maps_root': str(MAPS_ROOT),
          'cpu_only': True, 'read_only': True,
          'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                            for name in ['server_map_catalog_v1.py', 'launch_map_catalog_v1.py']}}
destination = audit / 'MAP_CATALOG_OWNER_V1.json'
temporary = destination.with_suffix('.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(destination)
print(json.dumps(record), flush=True)
server.serve_forever()

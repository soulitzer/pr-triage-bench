"""Run the pipeline's build_ownership_input.py with a snapshot of the PR's files.

Usage: python build_input.py <pipeline scripts dir> <files.json> <build args...>
Run as a script, not with -m, so the pipeline modules import as they do in CI.
"""

import json
import sys
from pathlib import Path

pipeline_dir, files_path = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], *sys.argv[3:]]
sys.path.insert(0, pipeline_dir)

import build_ownership_input  # noqa: E402

snapshot_files = json.loads(Path(files_path).read_text())
build_ownership_input.fetch_pull_request_files = lambda pr, /: snapshot_files
raise SystemExit(build_ownership_input.main())

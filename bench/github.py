"""Read-only GitHub access through the gh CLI."""

from __future__ import annotations

import json
import subprocess
from typing import Any


REPOSITORY = "pytorch/pytorch"


def gh_api(
    path: str,
    /,
    *,
    params: dict[str, str] | None = None,
    raw: bool = False,
    paginate: bool = False,
) -> Any:
    """Return a GitHub REST response as JSON, or as bytes when raw is set."""

    command = ["gh", "api", path]
    if params:
        command += ["-X", "GET"]
        for key, value in params.items():
            command += ["-f", f"{key}={value}"]
    if raw:
        command += ["-H", "Accept: application/vnd.github.raw"]
    if paginate:
        command += ["--paginate", "--slurp"]
    output = subprocess.run(command, check=True, capture_output=True).stdout
    if raw:
        return output
    value = json.loads(output)
    if paginate:
        return [item for page in value for item in (page if isinstance(page, list) else [page])]
    return value


def gh_token() -> str:
    return subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch_run_log(run_id: int, /) -> str:
    """Return the plain-text log of one workflow run."""

    return subprocess.run(
        ["gh", "run", "view", str(run_id), "-R", REPOSITORY, "--log"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

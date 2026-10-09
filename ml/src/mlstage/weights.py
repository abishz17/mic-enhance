"""Pinned weight downloads (direct URLs from official repos, sha256-verified)."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("mlstage")

MODELS_DIR = Path(os.environ.get("MLSTAGE_MODELS", Path(__file__).resolve().parents[2] / "models"))


def fetch(rel: str, url: str, sha256: str) -> Path:
    path = MODELS_DIR / rel
    stamp = path.with_name(path.name + ".verified")
    if path.is_file():
        st = path.stat()
        key = f"{sha256} {st.st_size} {st.st_mtime_ns}"
        if stamp.is_file() and stamp.read_text() == key:  # hashed before, file untouched since
            return path
        if _sha(path) == sha256:
            stamp.write_text(key)
            return path
        raise RuntimeError(f"{path} exists but its sha256 does not match the pinned value; delete it to re-download")
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - pinned https URL
    got = _sha(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: got {got}, want {sha256}")
    tmp.replace(path)
    return path


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

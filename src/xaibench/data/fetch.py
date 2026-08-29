"""Download, verify, and vendor the three source datasets.

All three URLs were verified reachable on 2026-08-29. Checksums are recorded on
first fetch and asserted on every later load: a silently changed upstream source
would invalidate every number in the paper, so it fails loudly instead.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass

from xaibench.config import paths


@dataclass(frozen=True)
class Source:
    dataset_id: str
    url: str
    licence: str
    citation: str


SOURCES = {
    "pima": Source(
        "pima",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        "CC0 1.0 (public domain); originally collected by the US National Institute "
        "of Diabetes and Digestive and Kidney Diseases",
        "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. "
        "Johannes, 'Using the ADAP learning algorithm to forecast the onset of "
        "diabetes mellitus,' in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, "
        "pp. 261-265.",
    ),
    "heart": Source(
        "heart",
        "https://archive.ics.uci.edu/static/public/45/data.csv",
        "CC BY 4.0 (UCI Machine Learning Repository, dataset 45)",
        "R. Detrano et al., 'International application of a new probability "
        "algorithm for the diagnosis of coronary artery disease,' Am. J. Cardiol., "
        "vol. 64, no. 5, pp. 304-310, 1989. UCI ML Repository, doi:10.24432/C52P4X.",
    ),
    "ckd": Source(
        "ckd",
        "https://archive.ics.uci.edu/static/public/336/data.csv",
        "CC BY 4.0 (UCI Machine Learning Repository, dataset 336)",
        "L. J. Rubini, P. Soundarapandian, and P. Eswaran, 'Chronic Kidney Disease,' "
        "UCI Machine Learning Repository, 2015, doi:10.24432/C5G020.",
    ),
}


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_all(force: bool = False) -> dict[str, str]:
    p = paths()
    p.mkdirs()
    lock_file = p.data_raw / "checksums.json"
    recorded = json.loads(lock_file.read_text()) if lock_file.exists() else {}
    digests: dict[str, str] = {}

    for ds_id, src in SOURCES.items():
        dest = p.data_raw / f"{ds_id}.csv"
        if force or not dest.exists():
            print(f"[fetch] {ds_id} <- {src.url}")
            request = urllib.request.Request(src.url, headers={"User-Agent": "xaibench/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                dest.write_bytes(response.read())
        digest = _sha256(dest)
        if ds_id in recorded and recorded[ds_id] != digest:
            raise RuntimeError(
                f"{ds_id}: SHA-256 mismatch.\n"
                f"  recorded {recorded[ds_id]}\n  found    {digest}\n"
                "The upstream source changed. Every result derived from it is now "
                "suspect. Investigate before rerunning; do not overwrite "
                "checksums.json to silence this."
            )
        digests[ds_id] = digest
        print(f"[fetch] {ds_id}: {dest.stat().st_size:,} bytes  sha256={digest[:16]}...")

    if not recorded:
        lock_file.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
        print(f"[fetch] recorded checksums -> {lock_file}")
    return digests


if __name__ == "__main__":
    fetch_all()

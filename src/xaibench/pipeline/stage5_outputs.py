"""Stage 5: figures, LaTeX tables, and the reproducibility manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time

from xaibench.config import get_profile, paths
from xaibench.stats.report import TABLES
from xaibench.viz.figures import build_all


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=paths().root).stdout.strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict[str, str]:
    import importlib.metadata as metadata
    wanted = ["numpy", "pandas", "scikit-learn", "scipy", "shap", "lime", "xgboost",
              "lightgbm", "statsmodels", "matplotlib", "pyarrow", "pyyaml"]
    out = {}
    for name in wanted:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = "absent"
    return out


def _hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor() or "unknown"}
    try:
        import os
        info["cpu_count"] = os.cpu_count()
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal"):
                    info["memory"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return info


def write_manifest(profile, duration_s: float) -> dict:
    p = paths()
    artefacts = {}
    for folder in (p.results, p.figures, p.tables):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                artefacts[str(path.relative_to(p.root))] = {
                    "sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "profile": profile.name,
        "reportable": profile.is_reportable,
        "git_commit": _git_commit(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(float(duration_s), 1),
        "seeds": {"global": 42, "repeat_r": "1000 + r", "bootstrap_b": "2000 + b",
                  "neighbour": "42 + instance_id"},
        "profile_settings": {k: v for k, v in vars(profile).items()},
        "library_versions": _library_versions(),
        "hardware": _hardware(),
        "artifacts": artefacts,
    }
    (p.results / "MANIFEST.json").write_text(json.dumps(manifest, indent=2,
                                                        default=str) + "\n")
    return manifest


def main(force: bool = False) -> None:
    started = time.perf_counter()
    profile, p = get_profile(), paths()
    p.mkdirs()

    print("[stage5] figures", flush=True)
    built = build_all(profile)
    print(f"[stage5] {len(built)} figures -> {p.figures}")

    print("[stage5] tables", flush=True)
    for name, builder in TABLES.items():
        (p.tables / f"{name}.tex").write_text(builder(profile))
    print(f"[stage5] {len(TABLES)} tables -> {p.tables}")

    manifest = write_manifest(profile, time.perf_counter() - started)
    print(f"[stage5] manifest: {len(manifest['artifacts'])} artefacts, "
          f"commit {manifest['git_commit'][:8]}, profile {manifest['profile']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))

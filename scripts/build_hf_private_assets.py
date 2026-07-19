"""Build the private, zero-image runtime assets for the hosted demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_package import _create_sanitized_catalog


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_safe_target(path):
    target = path.resolve()
    expected = (ROOT / "hf_private_assets").resolve()
    if target != expected:
        raise ValueError(f"Private asset target must be {expected}")
    return target


def build(model_path, database_path, output_directory):
    model_path = Path(model_path).resolve()
    database_path = Path(database_path).resolve()
    output_directory = ensure_safe_target(Path(output_directory))
    if not model_path.is_file() or not database_path.is_file():
        raise FileNotFoundError("The local model or database is missing.")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name in ("model_base.pth", "catalog.db", "ASSET_MANIFEST.json", "README.md"):
        (output_directory / name).unlink(missing_ok=True)

    target_model = output_directory / "model_base.pth"
    target_catalog = output_directory / "catalog.db"
    shutil.copy2(model_path, target_model)
    counts = _create_sanitized_catalog(database_path, target_catalog)

    connection = sqlite3.connect(target_catalog)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        path_count = connection.execute(
            "SELECT COUNT(*) FROM images WHERE orig_path IS NOT NULL OR seg_path IS NOT NULL"
        ).fetchone()[0]
    finally:
        connection.close()
    if integrity != "ok" or image_count != 0 or path_count != 0:
        raise RuntimeError("Sanitized catalog validation failed.")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "TemperamentOmicsNet",
        "embedding_dimension": 128,
        "attention_heads": 4,
        "images_included": False,
        "image_paths_included": False,
        "training_checkpoint_included": False,
        "catalog": counts,
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (target_model, target_catalog)
        },
    }
    (output_directory / "ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "README.md").write_text(
        """# Haworthia OMICS private runtime assets

This private repository is used only by the hosted inference service. It contains a
trained state dictionary and a zero-image numerical prototype catalog. It contains no
training images, gallery images, image paths, database image rows, or checkpoints.

The repository is not a public model release and grants no permission to download or
redistribute the model artifacts. Public application source is available at
https://github.com/YujunCC/haworthia-omics under Apache-2.0.
""",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ROOT / "model_base.pth", type=Path)
    parser.add_argument("--database", default=ROOT / "haworthia_omics.db", type=Path)
    parser.add_argument(
        "--output", default=ROOT / "hf_private_assets", type=Path
    )
    args = parser.parse_args()
    manifest = build(args.model, args.database, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

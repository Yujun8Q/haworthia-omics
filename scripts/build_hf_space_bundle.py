"""Build and audit the public Hugging Face Space source directory."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hf_space"
TARGET = ROOT / "hf_space_bundle"
TEMPLATE_FILES = {
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "app.py",
    "download_segmentation_models.py",
    "inference_core.py",
    "model.py",
    "requirements.txt",
    "runtime_assets.py",
}
ROOT_FILES = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "segmentation.py",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".db", ".jpeg", ".jpg", ".onnx", ".png", ".pt", ".pth", ".sqlite",
}


def safe_recreate_target():
    target = TARGET.resolve()
    if target != (ROOT / "hf_space_bundle").resolve():
        raise RuntimeError("Unexpected Hugging Face bundle target.")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def main():
    missing = [name for name in TEMPLATE_FILES if not (SOURCE / name).is_file()]
    missing += [name for name in ROOT_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing public Space source: " + ", ".join(sorted(missing)))

    target = safe_recreate_target()
    for name in sorted(TEMPLATE_FILES):
        shutil.copy2(SOURCE / name, target / name)
    for name in sorted(ROOT_FILES):
        shutil.copy2(ROOT / name, target / name)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "public_space_source": True,
        "model_weights_included": False,
        "numeric_prototypes_included": False,
        "images_included": False,
        "database_included": False,
        "training_or_maintenance_routes_included": False,
        "files": sorted(path.name for path in target.iterdir()),
    }
    (target / "PUBLIC_BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    violations = []
    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(path.name)
    if violations:
        raise RuntimeError("Forbidden public bundle files: " + ", ".join(violations))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

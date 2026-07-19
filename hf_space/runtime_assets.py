import hashlib
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED_FILES = ("ASSET_MANIFEST.json", "model_base.pth", "catalog.db")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_assets(directory):
    manifest_path = directory / "ASSET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported private asset schema.")
    if manifest.get("images_included") is not False:
        raise RuntimeError("Private assets must not contain image records.")
    for name in ("model_base.pth", "catalog.db"):
        metadata = manifest.get("files", {}).get(name, {})
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Missing private asset: {name}")
        if path.stat().st_size != metadata.get("size_bytes"):
            raise RuntimeError(f"Private asset size mismatch: {name}")
        if sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"Private asset checksum mismatch: {name}")
    return manifest


def prepare_private_assets(repo_id=None, token=None):
    local_override = os.getenv("HAWORTHIA_ASSET_DIR", "").strip()
    if local_override:
        directory = Path(local_override).expanduser().resolve()
        return directory, _validate_assets(directory)

    repo_id = (repo_id or os.getenv("HF_MODEL_REPO_ID", "")).strip()
    token = (token or os.getenv("HF_TOKEN", "")).strip()
    if not repo_id or not token:
        raise RuntimeError(
            "HF_MODEL_REPO_ID and a read-only HF_TOKEN are required."
        )
    default_directory = Path(tempfile.gettempdir()) / "haworthia-assets"
    directory = Path(os.getenv("HAWORTHIA_RUNTIME_DIR", str(default_directory)))
    directory.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            token=token,
            local_dir=directory,
        )
    return directory, _validate_assets(directory)

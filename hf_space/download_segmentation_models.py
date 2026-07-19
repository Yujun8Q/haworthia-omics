import hashlib
import os
import urllib.request
from pathlib import Path


MODELS = {
    "isnet-general-use.onnx": {
        "url": (
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/isnet-general-use.onnx"
        ),
        "sha256": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
    },
    "u2net.onnx": {
        "url": (
            "https://github.com/danielgatis/rembg/releases/download/"
            "v0.0.0/u2net.onnx"
        ),
        "sha256": "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_segmentation_models(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, metadata in MODELS.items():
        target = destination / name
        if target.is_file() and sha256_file(target) == metadata["sha256"]:
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(metadata["url"], temporary)
            if sha256_file(temporary) != metadata["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_segmentation_models(os.getenv("U2NET_HOME", "~/.u2net"))

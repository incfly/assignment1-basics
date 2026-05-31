#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "../cs336-data"
DEFAULT_LOCATION = "US"
TINYSTORY_ARTIFACTS = (
    "tinystory/TinyStories-train.txt-tokenized.bin",
    "tinystory/TinyStories-valid.txt-tokenized.bin",
    "tinystory/TinyStories-train.txt-vocab.json",
    "tinystory/TinyStories-train.txt-merge.json",
    "tinystory/TinyStories-train.txt-tokenized.json",
    "tinystory/TinyStories-valid.txt-tokenized.json",
)
GCS_ACCESS_TOKEN_FILE = Path("/tmp/gcs-token")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shell_quote(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=check)


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/:=.,@%+-"
    if all(ch in safe for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing required tool: {name}")


def gcloud_access_token() -> str:
    require_tool("gcloud")
    proc = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        check=True,
        text=True,
        capture_output=True,
    )
    token = proc.stdout.strip()
    if not token:
        raise SystemExit("gcloud did not return an access token")
    return token


def refresh_gcs_access_token(path: Path = GCS_ACCESS_TOKEN_FILE) -> Path:
    token = gcloud_access_token()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    print(f"wrote GCS access token to {path}")
    return path


def gcloud_storage_cmd(*parts: str, use_access_token: bool = False) -> list[str]:
    cmd = ["gcloud"]
    if use_access_token and GCS_ACCESS_TOKEN_FILE.exists():
        cmd.append(f"--access-token-file={GCS_ACCESS_TOKEN_FILE}")
    cmd.extend(["storage", *parts])
    return cmd


def normalize_bucket(bucket: str) -> str:
    bucket = bucket.rstrip("/")
    return bucket if bucket.startswith("gs://") else f"gs://{bucket}"


def gs_join(bucket: str, *parts: str) -> str:
    clean = [part.strip("/") for part in parts if part and part.strip("/")]
    if not clean:
        return normalize_bucket(bucket)
    return normalize_bucket(bucket) + "/" + "/".join(clean)


def artifact_paths(root: Path) -> list[Path]:
    return [root / rel_path for rel_path in TINYSTORY_ARTIFACTS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: list[Path], root: Path, out: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"files": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", [])
    if not isinstance(files, list):
        raise SystemExit(f"invalid manifest: {path}")
    return files


def cmd_create_bucket(args: argparse.Namespace) -> None:
    require_tool("gcloud")
    bucket = normalize_bucket(args.bucket)
    if run(["gcloud", "storage", "buckets", "describe", bucket], check=False).returncode == 0:
        print(f"bucket already exists: {bucket}")
        return

    cmd = [
        "gcloud",
        "storage",
        "buckets",
        "create",
        bucket,
        "--project",
        args.project,
        "--location",
        args.location,
        "--uniform-bucket-level-access",
    ]
    run(cmd)


def cmd_manifest(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    paths = artifact_paths(root) if args.profile == "tinystory-tokenized" else sorted(p for p in root.rglob("*") if p.is_file())
    write_manifest(paths, root, Path(args.out).expanduser())


def cmd_upload(args: argparse.Namespace) -> None:
    require_tool("gcloud")
    root = Path(args.source).expanduser().resolve()
    bucket = normalize_bucket(args.bucket)
    destination = gs_join(bucket, args.prefix)

    if args.all:
        run(["gcloud", "storage", "rsync", str(root), destination, "--recursive"])
        return

    for path in artifact_paths(root):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        rel = path.relative_to(root).as_posix()
        run(["gcloud", "storage", "cp", str(path), gs_join(bucket, args.prefix, rel)])


def cmd_download(args: argparse.Namespace) -> None:
    bucket = normalize_bucket(args.bucket)
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.signed_manifest:
        require_tool("aria2c")
        files = read_manifest(Path(args.signed_manifest).expanduser())
        for item in files:
            rel = item["path"]
            url = item["url"]
            out_dir = dest / str(Path(rel).parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            run(["aria2c", "-x", str(args.connections), "-s", str(args.connections), "-c", "-d", str(out_dir), "-o", Path(rel).name, url])
        return

    require_tool("gcloud")
    source = gs_join(bucket, args.prefix)
    run(gcloud_storage_cmd("rsync", source, str(dest), "--recursive", use_access_token=True))


def cmd_sign(args: argparse.Namespace) -> None:
    require_tool("gcloud")
    manifest_path = Path(args.manifest).expanduser()
    files = read_manifest(manifest_path)
    bucket = normalize_bucket(args.bucket)
    signed: list[dict[str, Any]] = []
    for item in files:
        rel = item["path"]
        url = gs_join(bucket, args.prefix, rel)
        proc = subprocess.run(
            ["gcloud", "storage", "sign-url", url, "--duration", args.duration, "--format", "json"],
            check=True,
            text=True,
            capture_output=True,
        )
        signed_url = _extract_signed_url(json.loads(proc.stdout))
        if not signed_url:
            print(proc.stdout, file=sys.stderr)
            raise SystemExit(f"could not parse signed URL for {url}")
        signed.append({**item, "url": signed_url})
    out = Path(args.out).expanduser()
    out.write_text(json.dumps({"files": signed}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _extract_signed_url(value: Any) -> str:
    if isinstance(value, str):
        return value if value.startswith("https://") else ""
    if isinstance(value, dict):
        for key, child in value.items():
            if "url" in key.lower():
                found = _extract_signed_url(child)
                if found:
                    return found
        for child in value.values():
            found = _extract_signed_url(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _extract_signed_url(child)
            if found:
                return found
    return ""


def cmd_verify(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    failures = 0
    for item in read_manifest(Path(args.manifest).expanduser()):
        path = root / item["path"]
        if not path.exists():
            print(f"missing {path}")
            failures += 1
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(item["size"]) or digest != item["sha256"]:
            print(f"mismatch {path}: size={size} sha256={digest}")
            failures += 1
        else:
            print(f"ok {path}")
    if failures:
        raise SystemExit(f"{failures} verification failure(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage CS336 data artifacts in object storage.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-bucket", help="create a GCS bucket if it does not already exist")
    p.add_argument("--bucket", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--location", default=DEFAULT_LOCATION)
    p.set_defaults(func=cmd_create_bucket)

    p = sub.add_parser("manifest", help="write a size/sha256 manifest")
    p.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--profile", choices=["tinystory-tokenized", "all"], default="tinystory-tokenized")
    p.add_argument("--out", default="artifacts/tinystory-tokenized-manifest.json")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("upload", help="upload data to GCS")
    p.add_argument("--source", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--all", action="store_true", help="sync the entire source directory instead of curated artifacts")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("download", help="download data from GCS or signed URLs")
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--dest", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--signed-manifest", default=None)
    p.add_argument("--connections", type=int, default=int(os.environ.get("ARIA2_CONNECTIONS", "16")))
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("sign", help="create a signed-URL manifest for aria2c downloads")
    p.add_argument("--manifest", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--duration", default="12h")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_sign)

    p = sub.add_parser("verify", help="verify files against a manifest")
    p.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

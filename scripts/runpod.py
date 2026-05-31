#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_FILES = (
    "TinyStories-train.txt-vocab.json",
    "TinyStories-train.txt-merge.json",
    "TinyStories-train.txt-tokenized.bin",
    "TinyStories-train.txt-tokenized.json",
    "TinyStories-valid.txt-tokenized.bin",
    "TinyStories-valid.txt-tokenized.json",
)


@dataclass(frozen=True)
class Target:
    user: str
    host: str
    port: str
    key: str

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}"

    def ssh_args(self) -> list[str]:
        args = [
            "ssh",
            "-p",
            self.port,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={Path.home() / '.ssh/runpod_known_hosts'}",
        ]
        if self.key != "-":
            args.extend(["-i", self.key])
        args.append(self.address)
        return args

    def rsync_ssh(self) -> str:
        return " ".join(shlex.quote(part) for part in self.ssh_args()[:-1])

    def scp_args(self) -> list[str]:
        args = [
            "scp",
            "-P",
            self.port,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={Path.home() / '.ssh/runpod_known_hosts'}",
        ]
        if self.key != "-":
            args.extend(["-i", self.key])
        return args


def cred_path() -> Path:
    return Path(os.environ.get("RUNPOD_CRED_FILE", "~/workspace/creds/runpods-cred.txt")).expanduser()


def read_cred() -> str:
    path = cred_path()
    if not path.exists():
        raise SystemExit(f"missing credential file: {path}")
    return path.read_text(encoding="utf-8").strip()


def read_env_cred(text: str) -> dict[str, str]:
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def api_key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    text = read_cred()
    env = read_env_cred(text) if "\n" in text or "=" in text else {}
    if env.get("RUNPOD_API_KEY"):
        return env["RUNPOD_API_KEY"]
    return text


def pod_id_file() -> Path:
    return Path(os.environ.get("RUNPOD_POD_ID_FILE", REPO_ROOT / ".runpod_pod_id")).expanduser()


def saved_pod_id() -> str | None:
    path = pod_id_file()
    if path.exists():
        pod_id = path.read_text(encoding="utf-8").strip()
        if pod_id:
            return pod_id
    return None


def write_pod_id(pod_id: str) -> None:
    pod_id_file().write_text(pod_id + "\n", encoding="utf-8")


def default_key() -> str:
    key = os.environ.get("RUNPOD_KEY")
    if key:
        return str(Path(key).expanduser())
    default = Path("~/.ssh/id_ed25519").expanduser()
    return str(default) if default.exists() else "-"


def parse_ssh_command(text: str) -> Target:
    tokens = shlex.split(text)
    looks_like_ssh = tokens[:1] == ["ssh"] or any("@" in token for token in tokens) or "-p" in tokens
    if not looks_like_ssh:
        raise ValueError("not an SSH command")
    if tokens[:1] == ["ssh"]:
        tokens = tokens[1:]

    user = os.environ.get("RUNPOD_USER", "root")
    host = ""
    port = os.environ.get("RUNPOD_PORT", "22")
    key = default_key()
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-p" and idx + 1 < len(tokens):
            port = tokens[idx + 1]
            idx += 2
            continue
        if token == "-i" and idx + 1 < len(tokens):
            key = str(Path(tokens[idx + 1]).expanduser())
            idx += 2
            continue
        if "@" in token:
            user, host = token.split("@", 1)
        elif not token.startswith("-") and not host:
            host = token
        idx += 1

    if not host:
        raise ValueError("no host found in SSH command")
    return Target(user=user, host=host, port=port, key=key)


def query_pods(api_key: str) -> dict:
    query = """
    query {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime {
            ports {
              ip
              privatePort
              publicPort
              type
            }
          }
        }
      }
    }
    """
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key}",
        data=payload,
        headers={"content-type": "application/json", "user-agent": "curl/8.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_from_api_key(api_key: str) -> Target:
    payload = query_pods(api_key)
    if payload.get("errors"):
        raise SystemExit(f"RunPod API error: {payload['errors']}")

    target_pod = os.environ.get("RUNPOD_POD_ID") or saved_pod_id()
    pods = payload.get("data", {}).get("myself", {}).get("pods", [])
    for pod in pods:
        if target_pod and pod.get("id") != target_pod:
            continue
        runtime = pod.get("runtime") or {}
        for port in runtime.get("ports") or []:
            if int(port.get("privatePort") or 0) == 22 and port.get("ip") and port.get("publicPort"):
                return Target(
                    user=os.environ.get("RUNPOD_USER", "root"),
                    host=str(port["ip"]),
                    port=str(port["publicPort"]),
                    key=default_key(),
                )
    raise SystemExit("no running pod with an exposed SSH port found")


def resolve_target() -> Target:
    if os.environ.get("RUNPOD_SSH_CMD"):
        return parse_ssh_command(os.environ["RUNPOD_SSH_CMD"])
    if os.environ.get("RUNPOD_HOST"):
        return Target(
            user=os.environ.get("RUNPOD_USER", "root"),
            host=os.environ["RUNPOD_HOST"],
            port=os.environ.get("RUNPOD_PORT", "22"),
            key=default_key(),
        )

    text = read_cred()
    if "\n" in text or "=" in text:
        env = read_env_cred(text)
        if "RUNPOD_HOST" in env:
            return Target(
                user=env.get("RUNPOD_USER", "root"),
                host=env["RUNPOD_HOST"],
                port=env.get("RUNPOD_PORT", "22"),
                key=str(Path(env.get("RUNPOD_KEY", default_key())).expanduser()),
            )

    try:
        return parse_ssh_command(text)
    except ValueError:
        return resolve_from_api_key(api_key())


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env)


def run_capture(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    result = subprocess.run(cmd, check=False, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        raise SystemExit(result.returncode)
    return result


def load_blob_manager():
    path = SCRIPT_DIR / "blob-manager.py"
    spec = importlib.util.spec_from_file_location("blob_manager", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_gcs_token(target: Target) -> None:
    blob_manager = load_blob_manager()
    local_path = blob_manager.refresh_gcs_access_token()
    remote_path = str(blob_manager.GCS_ACCESS_TOKEN_FILE)
    remote(target, f"mkdir -p {shlex.quote(str(Path(remote_path).parent))}")
    run([*target.scp_args(), str(local_path), f"{target.address}:{remote_path}"])
    remote(target, f"chmod 600 {shlex.quote(remote_path)}")


def extract_pod_id(value) -> str | None:
    if isinstance(value, dict):
        for key in ("id", "podId"):
            found = value.get(key)
            if isinstance(found, str) and found:
                return found
        for child in value.values():
            found = extract_pod_id(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = extract_pod_id(child)
            if found:
                return found
    return None


def remote(target: Target, command: str) -> None:
    run([*target.ssh_args(), command])


def ensure_remote_dir(target: Target, path: str) -> None:
    remote(target, f"mkdir -p {shlex.quote(path)} && (command -v rsync >/dev/null || (apt-get update && apt-get install -y rsync))")


def rsync(target: Target, src: str, dest: str, args: argparse.Namespace, extra: list[str] | None = None) -> None:
    cmd = ["rsync", "-avz", "--no-owner", "--no-group", "--progress"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.delete:
        cmd.append("--delete")
    if extra:
        cmd.extend(extra)
    cmd.extend(["-e", target.rsync_ssh(), src, dest])
    run(cmd)


def selected_parts(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    flags = [args.code_only, args.data_only, args.artifacts_only]
    if sum(bool(flag) for flag in flags) > 1:
        raise SystemExit("choose only one of --code-only, --data-only, or --artifacts-only")
    if args.code_only:
        return True, False, False
    if args.data_only:
        return False, True, False
    if args.artifacts_only:
        return False, False, True
    return True, True, False


def sync_code(target: Target, mode: str, args: argparse.Namespace) -> None:
    remote_dir = args.remote_dir
    excludes = [
        f"--exclude-from={REPO_ROOT / '.rsyncignore'}",
        "--exclude=/runs/",
        "--exclude=/checkpoints/",
        "--exclude=/logs/",
    ]
    if mode == "push":
        ensure_remote_dir(target, remote_dir)
        rsync(target, f"{REPO_ROOT}/", f"{target.address}:{remote_dir}/", args, excludes)
    else:
        Path(args.local_repo_dir).mkdir(parents=True, exist_ok=True)
        rsync(target, f"{target.address}:{remote_dir}/", f"{args.local_repo_dir}/", args, excludes)


def sync_data(target: Target, mode: str, args: argparse.Namespace) -> None:
    local_dir = Path(args.local_data_dir)
    remote_dir = args.remote_data_dir
    files = args.data_files or list(DEFAULT_DATA_FILES)
    if mode == "push":
        ensure_remote_dir(target, remote_dir)
        for file in files:
            rsync(target, str(local_dir / file), f"{target.address}:{remote_dir}/", args)
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            rsync(target, f"{target.address}:{remote_dir}/{file}", f"{local_dir}/", args)


def sync_artifacts(target: Target, mode: str, args: argparse.Namespace) -> None:
    if mode == "push":
        raise SystemExit("artifact sync is pull-only")
    local_root = Path(args.local_repo_dir)
    remote_dir = args.remote_dir
    local_root.mkdir(parents=True, exist_ok=True)
    (local_root / "runs").mkdir(parents=True, exist_ok=True)
    (local_root / "logs").mkdir(parents=True, exist_ok=True)
    run_ids = args.run_id or []
    if run_ids:
        for run_id in run_ids:
            rsync(target, f"{target.address}:{remote_dir}/runs/{run_id}/", f"{local_root}/runs/{run_id}/", args)
            rsync(target, f"{target.address}:{remote_dir}/logs/{run_id}.log", f"{local_root}/logs/", args)
    else:
        rsync(target, f"{target.address}:{remote_dir}/runs/", f"{local_root}/runs/", args)
        rsync(target, f"{target.address}:{remote_dir}/logs/", f"{local_root}/logs/", args)


def cmd_resolve(_args: argparse.Namespace) -> None:
    target = resolve_target()
    print(target.user, target.host, target.port, target.key)


def cmd_create(args: argparse.Namespace) -> None:
    if not shutil.which("runpodctl"):
        raise SystemExit("runpodctl not found; install it or create the pod from the RunPod console")
    if args.image and args.template_id:
        raise SystemExit("choose only one of --image or --template-id")

    env = os.environ.copy()
    env["RUNPOD_API_KEY"] = api_key()
    cmd = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        args.name,
        "--gpu-id",
        args.gpu_id,
        "--gpu-count",
        str(args.gpu_count),
        "--cloud-type",
        args.cloud_type,
        "--container-disk-in-gb",
        str(args.container_disk_in_gb),
        "--volume-in-gb",
        str(args.volume_in_gb),
        "--volume-mount-path",
        args.volume_mount_path,
        "--ports",
        args.ports,
        "-o",
        "json",
    ]
    if args.image:
        cmd.extend(["--image", args.image])
    else:
        cmd.extend(["--template-id", args.template_id])
    if args.data_center_ids:
        cmd.extend(["--data-center-ids", args.data_center_ids])
    if args.public_ip:
        cmd.append("--public-ip")
    if args.global_networking:
        cmd.append("--global-networking")

    result = run_capture(cmd, env=env)
    if result.stderr:
        print(result.stderr, end="")
    print(result.stdout, end="")

    pod_id = None
    try:
        pod_id = extract_pod_id(json.loads(result.stdout))
    except json.JSONDecodeError:
        pass
    if pod_id:
        write_pod_id(pod_id)
        print(f"saved pod id {pod_id} to {pod_id_file()}")

    if not args.wait_ssh:
        return

    deadline = time.time() + args.wait_ssh_timeout
    while time.time() < deadline:
        try:
            target = resolve_from_api_key(env["RUNPOD_API_KEY"])
            print(f"ssh ready: {target.user} {target.host} {target.port} {target.key}")
            return
        except SystemExit as exc:
            print(f"waiting for ssh: {exc}")
            time.sleep(args.wait_ssh_interval)
    raise SystemExit("timed out waiting for SSH; the pod may still be starting")


def cmd_ssh(args: argparse.Namespace) -> None:
    target = resolve_target()
    command = " ".join(shlex.quote(part) for part in args.command)
    cmd = target.ssh_args()
    if command:
        cmd.append(command)
    raise SystemExit(subprocess.call(cmd))


def cmd_sync(args: argparse.Namespace) -> None:
    target = resolve_target()
    code, data, artifacts = selected_parts(args)
    if code:
        sync_code(target, args.mode, args)
    if data:
        sync_data(target, args.mode, args)
    if artifacts:
        sync_artifacts(target, args.mode, args)


def cmd_setup(args: argparse.Namespace) -> None:
    target = resolve_target()
    args.mode = "push"
    if args.artifacts_only:
        raise SystemExit("--artifacts-only does not apply to setup")
    if not args.no_sync:
        code, data, _artifacts = selected_parts(args)
        if code:
            sync_code(target, "push", args)
        if data:
            sync_data(target, "push", args)
    refresh_gcs_token(target)
    remote(target, f"cd {shlex.quote(args.remote_dir)} && ./scripts/setup_train_linux.sh")


def cmd_refresh_gcs_token(_args: argparse.Namespace) -> None:
    refresh_gcs_token(resolve_target())


def cmd_stop(args: argparse.Namespace) -> None:
    pod_id = args.pod_id or os.environ.get("RUNPOD_POD_ID") or saved_pod_id()
    if not pod_id:
        key = api_key()
        payload = query_pods(key)
        pods = payload.get("data", {}).get("myself", {}).get("pods", [])
        if len(pods) == 1:
            pod_id = pods[0].get("id")
        else:
            target = None
            try:
                target = resolve_target()
            except SystemExit:
                pass
            for pod in pods:
                if target:
                    runtime = pod.get("runtime") or {}
                    for port in runtime.get("ports") or []:
                        if str(port.get("ip")) == target.host and str(port.get("publicPort")) == target.port:
                            pod_id = pod.get("id")
                            break
                if pod_id:
                    break
                if pod.get("desiredStatus") == "RUNNING":
                    pod_id = pod.get("id")
                    break
    if not pod_id:
        raise SystemExit("pod id not found; pass --pod-id or set RUNPOD_POD_ID")

    env = os.environ.copy()
    env["RUNPOD_API_KEY"] = api_key()
    if not shutil.which("runpodctl"):
        raise SystemExit("runpodctl not found; install it or stop from the RunPod console")
    run(["runpodctl", "pod", "stop", pod_id, "-o", "json"], env=env)


def add_sync_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-only", action="store_true", help="sync only repository code")
    parser.add_argument("--data-only", action="store_true", help="sync only configured data files")
    parser.add_argument("--artifacts-only", action="store_true", help="pull only runs/ and logs/")
    parser.add_argument("--data-file", action="append", dest="data_files", help="data filename to sync; repeatable")
    parser.add_argument("--remote-dir", default=os.environ.get("REMOTE_DIR", "/workspace/assignment1-basics"))
    parser.add_argument("--local-repo-dir", default=str(REPO_ROOT))
    parser.add_argument("--remote-data-dir", default=os.environ.get("REMOTE_DATA_DIR", "/workspace/cs336-data/tinystory"))
    parser.add_argument("--local-data-dir", default=os.environ.get("LOCAL_DATA_DIR", str(REPO_ROOT / "../cs336-data/tinystory")))
    parser.add_argument("--run-id", action="append", help="artifact run id to pull; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod helper for code, data, setup, and artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create a RunPod pod and remember its pod id")
    p.add_argument("--name", default=f"assignment1-basics-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--gpu-id", default=os.environ.get("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4070"))
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cloud-type", default=os.environ.get("RUNPOD_CLOUD_TYPE", "COMMUNITY"))
    p.add_argument("--template-id", default=os.environ.get("RUNPOD_TEMPLATE_ID", "runpod-torch-v280"))
    p.add_argument("--image", default=os.environ.get("RUNPOD_IMAGE"))
    p.add_argument("--data-center-ids", default=os.environ.get("RUNPOD_DATA_CENTER_IDS"))
    p.add_argument("--container-disk-in-gb", type=int, default=30)
    p.add_argument("--volume-in-gb", type=int, default=20)
    p.add_argument("--volume-mount-path", default="/workspace")
    p.add_argument("--ports", default="22/tcp")
    p.add_argument("--public-ip", action="store_true")
    p.add_argument("--global-networking", action="store_true")
    p.add_argument("--no-wait-ssh", action="store_false", dest="wait_ssh")
    p.add_argument("--wait-ssh-timeout", type=int, default=600)
    p.add_argument("--wait-ssh-interval", type=int, default=10)
    p.set_defaults(func=cmd_create, wait_ssh=True)

    p = sub.add_parser("resolve", help="print resolved user host port key")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("ssh", help="open ssh or run a remote command")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_ssh)

    for mode in ("push", "pull"):
        p = sub.add_parser(mode, help=f"{mode} code and data by default")
        p.set_defaults(func=cmd_sync, mode=mode)
        add_sync_flags(p)

    p = sub.add_parser("setup", help="push code/data, then run remote training setup")
    add_sync_flags(p)
    p.add_argument("--no-sync", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("refresh-gcs-token", help="refresh and copy the GCS token to the current RunPod pod")
    p.set_defaults(func=cmd_refresh_gcs_token)

    p = sub.add_parser("stop", help="stop the current RunPod pod")
    p.add_argument("--pod-id", default=None)
    p.set_defaults(func=cmd_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

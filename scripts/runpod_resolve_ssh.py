#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import sys
import urllib.request
from pathlib import Path


def _cred_path() -> Path:
    return Path(os.environ.get("RUNPOD_CRED_FILE", "~/workspace/creds/runpods-cred.txt")).expanduser()


def _read_cred() -> str:
    path = _cred_path()
    if not path.exists():
        raise SystemExit(f"missing credential file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _default_key() -> str:
    if os.environ.get("RUNPOD_KEY"):
        return os.environ["RUNPOD_KEY"]
    key = Path("~/.ssh/id_ed25519").expanduser()
    return str(key) if key.exists() else "-"


def _parse_ssh_command(text: str) -> tuple[str, str, str, str]:
    tokens = shlex.split(text)
    looks_like_ssh = tokens[:1] == ["ssh"] or any("@" in token for token in tokens) or "-p" in tokens
    if not looks_like_ssh:
        raise ValueError("not an SSH command")

    if tokens[:1] == ["ssh"]:
        tokens = tokens[1:]

    user = os.environ.get("RUNPOD_USER", "root")
    host = ""
    port = os.environ.get("RUNPOD_PORT", "22")
    key = _default_key()
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
    return user, host, port, key


def _query_pods(api_key: str) -> dict:
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


def _resolve_from_api_key(api_key: str) -> tuple[str, str, str, str]:
    payload = _query_pods(api_key)
    if payload.get("errors"):
        raise SystemExit(f"RunPod API error: {payload['errors']}")

    target_pod = os.environ.get("RUNPOD_POD_ID")
    pods = payload.get("data", {}).get("myself", {}).get("pods", [])
    for pod in pods:
        if target_pod and pod.get("id") != target_pod:
            continue
        runtime = pod.get("runtime") or {}
        for port in runtime.get("ports") or []:
            if int(port.get("privatePort") or 0) == 22 and port.get("ip") and port.get("publicPort"):
                return (
                    os.environ.get("RUNPOD_USER", "root"),
                    str(port["ip"]),
                    str(port["publicPort"]),
                    _default_key(),
                )

    raise SystemExit("no running pod with an exposed SSH port found")


def main() -> None:
    if os.environ.get("RUNPOD_SSH_CMD"):
        print(*_parse_ssh_command(os.environ["RUNPOD_SSH_CMD"]))
        return

    if os.environ.get("RUNPOD_HOST"):
        print(
            os.environ.get("RUNPOD_USER", "root"),
            os.environ["RUNPOD_HOST"],
            os.environ.get("RUNPOD_PORT", "22"),
            _default_key(),
        )
        return

    text = _read_cred()
    if "\n" in text or "=" in text:
        env = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
        if "RUNPOD_HOST" in env:
            print(
                env.get("RUNPOD_USER", "root"),
                env["RUNPOD_HOST"],
                env.get("RUNPOD_PORT", "22"),
                env.get("RUNPOD_KEY", "-"),
            )
            return

    try:
        user, host, port, key = _parse_ssh_command(text)
    except ValueError:
        user, host, port, key = _resolve_from_api_key(text)
    print(user, host, port, key)


if __name__ == "__main__":
    main()

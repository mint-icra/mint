#!/usr/bin/env python3
"""Small, testable wrapper around the installed Aliyun DLC CLI."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Mapping

from .config import AliyunConfig


_JOB_ID_RE = re.compile(r"\bdlc[a-z0-9]{8,}\b", re.IGNORECASE)


class DlcError(RuntimeError):
    pass


def credential_environment(credentials: Path | str | None) -> dict[str, str]:
    """Source the existing shell credential file without exposing secrets in argv/logs."""
    env = os.environ.copy()
    if credentials is None:
        return env
    path = Path(credentials)
    if not path.is_file():
        return env
    proc = subprocess.run(
        ["bash", "-c", 'set -a; source "$1"; env -0', "aliyun-credentials", str(path)],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise DlcError(f"加载 Aliyun 凭证失败: {message or proc.returncode}")
    for item in proc.stdout.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def remote_command(config: AliyunConfig, request_path: Path | str) -> str:
    repo = shlex.quote(config.repo_dir)
    conda = shlex.quote(config.conda_env)
    request = shlex.quote(str(request_path))
    worker = shlex.quote(
        f"{config.repo_dir}/eval/model_effect/benchmark/dist/aliyun/worker.py"
    )
    script = (
        "set -euo pipefail; "
        "cd " + repo + "; "
        "exec conda run --no-capture-output -n " + conda + " python "
        + worker + " --request " + request
    )
    return "bash -lc " + shlex.quote(script)


def submission_name(config: AliyunConfig, _stamp: str) -> str:
    """Use the configured name verbatim so the DLC task is easy to find."""
    return config.job_name


def build_submit_args(config: AliyunConfig, request_path: Path | str,
                      *, display_name: str) -> list[str]:
    return [
        "dlc", "submit", "pytorchjob",
        "-e", config.endpoint, "-r", config.region, "-w", config.workspace_id,
        "--name", display_name,
        "--resource_id", config.resource_id,
        "--workers", str(config.nnodes),
        "--worker_image", config.image,
        "--worker_cpu", str(config.worker_cpu),
        "--worker_gpu", str(config.gpus_per_node),
        "--worker_memory", config.worker_memory,
        "--worker_shared_memory", config.worker_shared_memory,
        "--data_source_uris", f"{config.cpfs_uri}::/benchmark-data/",
        "--envs", (
            f"NNODES={config.nnodes},GPUS_PER_NODE={config.gpus_per_node},"
            "NCCL_SOCKET_IFNAME=eth0,NCCL_IB_DISABLE=0,NCCL_P2P_DISABLE=0"
        ),
        "--command", remote_command(config, request_path),
    ]


class DlcClient:
    def __init__(self, config: AliyunConfig, *, credentials: Path | str | None = None,
                 timeout: int = 30):
        self.config = config
        self.env = credential_environment(credentials)
        self.timeout = int(timeout)

    def _run(self, args: list[str], timeout: int | None = None) -> str:
        try:
            proc = subprocess.run(
                args, env=self.env, capture_output=True, text=True,
                timeout=timeout or self.timeout, check=False,
            )
        except FileNotFoundError as exc:
            raise DlcError("未找到 dlc CLI，请先安装或配置 PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise DlcError(f"dlc 命令超时（{timeout or self.timeout}s）") from exc
        output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        if proc.returncode != 0:
            raise DlcError(output or f"dlc 命令失败，returncode={proc.returncode}")
        return output

    def submit(self, request_path: Path | str, *, display_name: str) -> tuple[str, str]:
        output = self._run(
            build_submit_args(self.config, request_path, display_name=display_name),
            timeout=max(60, self.timeout),
        )
        match = _JOB_ID_RE.search(output)
        if match:
            return match.group(0), output
        # Some DLC CLI versions only print a success line. The unique display name lets us
        # resolve the newly-created job without relying on that unstable presentation format.
        for _attempt in range(6):
            time.sleep(2)
            listed = self._run([
                "dlc", "get", "job", "-e", self.config.endpoint,
                "-r", self.config.region, "-w", self.config.workspace_id,
                "-n", display_name, "--page_size", "10", "--order", "desc",
            ])
            match = _JOB_ID_RE.search(listed)
            if match:
                return match.group(0), output
        raise DlcError("作业已提交，但无法从 DLC 输出解析 JobId")

    def get_job(self, job_id: str) -> dict:
        output = self._run([
            "dlc", "get", "job", job_id,
            "-e", self.config.endpoint, "-r", self.config.region,
            "-w", self.config.workspace_id, "--show_detail",
        ])
        start = output.find("{")
        if start < 0:
            raise DlcError("DLC 作业详情不是 JSON")
        try:
            detail, _end = json.JSONDecoder().raw_decode(output[start:])
            return detail
        except json.JSONDecodeError as exc:
            raise DlcError(f"解析 DLC 作业详情失败: {exc}") from exc

    def stop(self, job_id: str) -> str:
        return self._run([
            "dlc", "stop", "job", job_id, "--force", "--quiet",
            "-e", self.config.endpoint, "-r", self.config.region,
        ], timeout=max(60, self.timeout))


def public_job_detail(detail: Mapping | None) -> dict:
    detail = detail or {}
    return {
        "job_id": detail.get("JobId"),
        "job_status": detail.get("Status"),
        "reason_code": detail.get("ReasonCode"),
        "reason_message": detail.get("ReasonMessage"),
        "created_at": detail.get("GmtCreateTime"),
        "submitted_at": detail.get("GmtSubmittedTime"),
        "running_at": detail.get("GmtRunningTime"),
        "finished_at": detail.get("GmtFinishTime"),
        "duration": detail.get("Duration"),
        "pods": [
            {
                "pod_id": pod.get("PodId"),
                "status": pod.get("Status"),
                "type": pod.get("Type"),
            }
            for pod in (detail.get("Pods") or [])
        ],
    }

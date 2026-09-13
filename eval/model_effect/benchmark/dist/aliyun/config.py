#!/usr/bin/env python3
"""Validated Aliyun Benchmark configuration shared by the Viewer and CLI."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_REGION_RE = re.compile(r"^[a-z0-9-]+$")
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:Mi|Gi|Ti)$")
_JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AliyunConfig:
    region: str
    workspace_id: str
    resource_id: str
    image: str
    cpfs_uri: str
    repo_dir: str
    nnodes: int
    gpus_per_node: int
    worker_cpu: int
    worker_memory: str
    worker_shared_memory: str
    conda_env: str
    job_name: str

    @property
    def world_size(self) -> int:
        return self.nnodes * self.gpus_per_node

    @property
    def endpoint(self) -> str:
        return f"pai-dlc.{self.region}.aliyuncs.com"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AliyunConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("aliyun 配置必须是对象")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError("未知 aliyun 配置项: " + ", ".join(unknown))

        missing = sorted(name for name in allowed if raw.get(name) in (None, ""))
        if missing:
            raise ValueError("缺少 aliyun 配置项: " + ", ".join(missing))
        try:
            config = cls(
                region=str(raw["region"]).strip(),
                workspace_id=str(raw["workspace_id"]).strip(),
                resource_id=str(raw["resource_id"]).strip(),
                image=str(raw["image"]).strip(),
                cpfs_uri=str(raw["cpfs_uri"]).strip(),
                repo_dir=str(raw["repo_dir"]).strip(),
                nnodes=int(raw["nnodes"]),
                gpus_per_node=int(raw["gpus_per_node"]),
                worker_cpu=int(raw["worker_cpu"]),
                worker_memory=str(raw["worker_memory"]).strip(),
                worker_shared_memory=str(raw["worker_shared_memory"]).strip(),
                conda_env=str(raw["conda_env"]).strip(),
                job_name=str(raw["job_name"]).strip(),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aliyun 数值配置无效: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not _REGION_RE.fullmatch(self.region):
            raise ValueError("region 只能包含小写字母、数字和连字符")
        for name, value in (
            ("workspace_id", self.workspace_id),
            ("resource_id", self.resource_id),
        ):
            if not _ID_RE.fullmatch(value):
                raise ValueError(f"{name} 含不支持的字符")
        for name, value in (
            ("image", self.image),
            ("repo_dir", self.repo_dir),
            ("conda_env", self.conda_env),
        ):
            if not value or any(ch in value for ch in "\r\n\0"):
                raise ValueError(f"{name} 不能为空或包含换行")
        if any(ch.isspace() for ch in self.image):
            raise ValueError("image 不能包含空白字符")
        if not self.cpfs_uri.startswith("bmcpfs://") or any(
            ch in self.cpfs_uri for ch in "\r\n\0"
        ):
            raise ValueError("cpfs_uri 必须是合法 bmcpfs:// URI")
        if not Path(self.repo_dir).is_absolute():
            raise ValueError("repo_dir 必须是绝对路径")
        if not 1 <= self.nnodes <= 128:
            raise ValueError("nnodes 必须在 1..128 之间")
        if not 1 <= self.gpus_per_node <= 8:
            raise ValueError("gpus_per_node 必须在 1..8 之间")
        if not 1 <= self.worker_cpu <= 1024:
            raise ValueError("worker_cpu 必须在 1..1024 之间")
        if not _MEMORY_RE.fullmatch(self.worker_memory):
            raise ValueError("worker_memory 格式应类似 1800Gi")
        if not _MEMORY_RE.fullmatch(self.worker_shared_memory):
            raise ValueError("worker_shared_memory 格式应类似 1800Gi")
        if not _JOB_RE.fullmatch(self.job_name) or len(self.job_name) > 180:
            raise ValueError("job_name 只能包含字母、数字、点、下划线、连字符，且不超过 180 字符")


def load_defaults(path: Path | str = DEFAULTS_PATH) -> AliyunConfig:
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    return AliyunConfig.from_mapping(document.get("aliyun") or {})

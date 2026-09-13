#!/usr/bin/env python3
"""Fail when first-party source exposes private infrastructure or identity clues."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".css", ".html", ".ini", ".js", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"
}
SKIP_PARTS = {".git", ".cache", "_vendor", "artifacts", "output", "third_party"}


@dataclass(frozen=True)
class Rule:
    name: str
    expression: re.Pattern[str]


RULES = (
    Rule("private absolute path", re.compile(r"(?<![A-Za-z0-9_])/(?:cpfs|mnt/(?:data|workspace)|root)(?:/|\b)", re.IGNORECASE)),
    Rule("private object storage", re.compile(r"\b(?:oss|s3)://[^\s'\"]+", re.IGNORECASE)),
    Rule("cloud access key", re.compile(r"\b(?:LTAI|AKLT)[A-Za-z0-9+/=]{12,}")),
    Rule("credential assignment", re.compile(r"(?i)\b(?:access[_-]?key|secret[_-]?key|password|api[_-]?key)\s*[:=]\s*['\"][^'\"]+")),
    Rule("private registry", re.compile(r"[A-Za-z0-9.-]+\.(?:cr|registry)\.[A-Za-z0-9.-]+/")),
    Rule("direct email address", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    Rule("personal contact channel", re.compile(r"(?i)\b(?:wechat|weixin)\b|微信|联系方式")),
    Rule("public preprint link", re.compile(r"(?i)arxiv\.org/(?:abs|pdf)/\d{4}\.\d+")),
)


def files_to_scan() -> list[Path]:
    return sorted(
        path
        for path in PROJECT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not any(part in SKIP_PARTS for part in path.relative_to(PROJECT_DIR).parts)
    )


def audit() -> list[tuple[Path, int, str, str]]:
    findings = []
    for path in files_to_scan():
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append((path, 0, "non-UTF-8 text", "File cannot be decoded as UTF-8"))
            continue
        for line_number, line in enumerate(lines, 1):
            for rule in RULES:
                match = rule.expression.search(line)
                if match:
                    excerpt = line.strip()[:160]
                    findings.append((path, line_number, rule.name, excerpt))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Return a non-zero status when findings exist.")
    args = parser.parse_args(argv)
    findings = audit()
    for path, line, rule, excerpt in findings:
        relative = path.relative_to(PROJECT_DIR)
        print(f"{relative}:{line}: {rule}: {excerpt}")
    print(f"Privacy audit: {len(findings)} finding(s) across {len(files_to_scan())} text files.")
    return 1 if args.strict and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

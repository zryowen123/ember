#!/usr/bin/env python
"""Merge EMBER raw-feature JSONL files with sha256 deduplication.

Example:
    python scripts/merge_features.py black1.jsonl black2.jsonl white.jsonl \
        --output merged.jsonl

By default, the same sha256 with a different label aborts the merge. Use
``--on-conflict keep-first`` to keep the first occurrence of such a sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_LABELS = {-1, 0, 1}


def source_text(obj: Dict[str, Any]) -> str:
    source = obj.get("source") or {}
    archive = source.get("archive") or "?"
    member = source.get("member") or "?"
    return f"{archive}:{member}"


def normalize_record(obj: Any, path: str, line_no: int) -> Tuple[str, int]:
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}:{line_no}: record is not a JSON object")
    sha = obj.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise SystemExit(f"{path}:{line_no}: missing or invalid sha256")
    try:
        int(sha, 16)
    except ValueError as exc:
        raise SystemExit(f"{path}:{line_no}: invalid sha256: {sha}") from exc
    label = obj.get("label")
    if isinstance(label, bool) or not isinstance(label, (int, float)):
        raise SystemExit(f"{path}:{line_no}: missing or invalid label")
    if float(label) not in VALID_LABELS:
        raise SystemExit(f"{path}:{line_no}: label must be -1, 0, or 1 (got {label})")
    return sha.lower(), int(label)


def merge(
    inputs: Sequence[str],
    output: str,
    on_conflict: str,
) -> int:
    input_paths = [os.path.abspath(path) for path in inputs]
    output_path = os.path.abspath(output)
    if output_path in input_paths:
        raise SystemExit("--output must not be one of the input files")
    for path in input_paths:
        if not os.path.exists(path):
            raise SystemExit(f"input does not exist: {path}")

    parent = os.path.dirname(output_path) or "."
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".merge_", suffix=".jsonl", dir=parent)
    os.close(fd)

    seen: Dict[str, Tuple[int, str]] = {}
    stats = {
        "files": len(input_paths),
        "read": 0,
        "kept": 0,
        "duplicates": 0,
        "label_conflicts": 0,
        "blank": 0,
        "label_-1": 0,
        "label_0": 0,
        "label_1": 0,
    }
    try:
        with open(tmp_path, "w", encoding="utf-8") as out:
            for path in input_paths:
                with open(path, encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if not line.strip():
                            stats["blank"] += 1
                            continue
                        stats["read"] += 1
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                        sha, label = normalize_record(obj, path, line_no)
                        location = f"{path}:{line_no} ({source_text(obj)})"
                        if sha in seen:
                            prev_label, prev_location = seen[sha]
                            if prev_label != label:
                                message = (
                                    f"label conflict for {sha}: previous label "
                                    f"{prev_label} at {prev_location}, new label "
                                    f"{label} at {location}"
                                )
                                if on_conflict == "error":
                                    raise SystemExit(
                                        message + "; use --on-conflict keep-first"
                                    )
                                print(f"WARNING: {message}; keeping first", file=sys.stderr)
                                stats["label_conflicts"] += 1
                            else:
                                stats["duplicates"] += 1
                            continue
                        seen[sha] = (label, location)
                        out.write(
                            json.dumps(
                                obj,
                                ensure_ascii=True,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        stats["kept"] += 1
                        stats[f"label_{label}"] += 1
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(
        "summary: "
        + " ".join(f"{key}={value}" for key, value in stats.items()),
        file=sys.stderr,
    )
    print(f"output: {output_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge EMBER feature JSONL files, deduplicating by sha256."
    )
    parser.add_argument("inputs", nargs="+", help="input JSONL files")
    parser.add_argument("--output", required=True, help="merged JSONL output")
    parser.add_argument(
        "--on-conflict",
        choices=("error", "keep-first"),
        default="error",
        help="what to do when the same sha256 has different labels (default: error)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return merge(args.inputs, args.output, args.on_conflict)


if __name__ == "__main__":
    raise SystemExit(main())

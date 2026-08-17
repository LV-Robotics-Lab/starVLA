#!/usr/bin/env python3
"""Thin, dependency-free Prometheus adapter for StarVLA.

The adapter validates repository structure and dispatches native entrypoints as
argv arrays.  It deliberately does not import StarVLA, install dependencies,
convert datasets, or authorize a checkpoint for robot control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = Path(__file__).with_name("capabilities.json")
TRAIN_ENTRYPOINT = Path("starVLA/training/train_starvla.py")
SERVE_ENTRYPOINT = Path("deployment/model_server/server_policy.py")
REQUIRED_PATHS = (
    TRAIN_ENTRYPOINT,
    SERVE_ENTRYPOINT,
    Path("requirements.txt"),
    Path("pyproject.toml"),
)


def capabilities() -> dict[str, object]:
    payload = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "prometheus_source_adapter_v1":
        raise RuntimeError("unsupported Prometheus source-adapter schema")
    return payload


def doctor() -> dict[str, object]:
    missing = [path.as_posix() for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    declared = capabilities()
    if declared["capabilities"]["resume"] != "weights_only":  # type: ignore[index]
        raise RuntimeError("StarVLA resume must remain declared as weights_only")
    if declared["capabilities"]["hardware_rollout_authorized"] is not False:  # type: ignore[index]
        raise RuntimeError("training source must not authorize hardware rollout")
    if missing:
        raise RuntimeError(f"missing required StarVLA paths: {missing}")
    return {
        "ok": True,
        "policy_id": declared["policy_id"],
        "checked_paths": [path.as_posix() for path in REQUIRED_PATHS],
        "imports_model_stack": False,
    }


def _native_args(values: Sequence[str]) -> list[str]:
    args = list(values)
    if args[:1] == ["--"]:
        args.pop(0)
    return args


def _require_config(args: Sequence[str]) -> None:
    if not any(arg == "--config_yaml" or arg.startswith("--config_yaml=") for arg in args):
        raise ValueError("StarVLA training requires an explicit --config_yaml")


def build_argv(
    stage: str,
    native_args: Sequence[str],
    *,
    accelerate_args: Sequence[str] = (),
    checkpoint: str | None = None,
) -> list[str]:
    """Build one native command without invoking a shell."""

    args = _native_args(native_args)
    if stage in {"train", "resume"}:
        _require_config(args)
        command = ["accelerate", "launch", *accelerate_args, TRAIN_ENTRYPOINT.as_posix(), *args]
        if stage == "resume":
            if not checkpoint:
                raise ValueError("weights-only resume requires --checkpoint")
            # StarVLA checkpoints contain model weights, not optimizer state.
            # Override is_resume so this adapter cannot imply full-state resume.
            command.extend(
                [
                    "--trainer.pretrained_checkpoint",
                    checkpoint,
                    "--trainer.is_resume",
                    "false",
                ]
            )
        return command
    if stage == "serve":
        return [sys.executable, SERVE_ENTRYPOINT.as_posix(), *args]
    raise ValueError(f"unsupported executable stage: {stage}")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _stage_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--plan", action="store_true", help="print argv instead of executing")
    parser.add_argument(
        "--accelerate-arg",
        action="append",
        default=[],
        help="one Accelerate launcher token; use --accelerate-arg=VALUE",
    )
    if name == "resume":
        parser.add_argument("--checkpoint", required=True)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capabilities")
    subparsers.add_parser("doctor")
    for stage in ("train", "resume", "serve"):
        _stage_parser(subparsers, stage)
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        _print_json(capabilities())
        return 0
    if args.command == "doctor":
        _print_json(doctor())
        return 0

    command = build_argv(
        args.command,
        args.native_args,
        accelerate_args=args.accelerate_arg,
        checkpoint=getattr(args, "checkpoint", None),
    )
    if args.plan:
        _print_json({"argv": command, "shell": False})
        return 0
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

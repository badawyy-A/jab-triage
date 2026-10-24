"""The jab command."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from jab_triage import __version__
from jab_triage.backends import KEY_VARS, from_env
from jab_triage.backends.base import Backend, BackendError
from jab_triage.presets import (
    PRESETS_DIR,
    Preset,
    PresetError,
    find_preset,
    list_presets,
    load_preset,
)
from jab_triage.triage import (
    DEFAULT_AUDIT_RATE,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_STATE_BYTES,
    NEEDS_REVIEW,
    InputError,
    parse_items,
    run,
    triage,
)

BackendFactory = Callable[[Mapping[str, str]], Backend]

# Exit code when a run stopped before sending because --yes was not given.
NEEDS_CONFIRMATION = 3


def main(argv: list[str] | None = None, backend_factory: BackendFactory = from_env) -> int:
    args = build_parser().parse_args(argv)

    # Our own handler, added per call and removed after, so nothing leaks into
    # other loggers. Item text is only ever logged at debug level.
    logger = logging.getLogger("jab_triage")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("jab: %(message)s"))
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if getattr(args, "debug", False) else logging.WARNING)
    try:
        return args.handler(args, backend_factory)
    except (PresetError, InputError, BackendError) as exc:
        print(f"jab: error: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jab", description="Sort items with Jev and review only the uncertain ones."
    )
    parser.add_argument("--version", action="version", version=f"jab {__version__}")
    commands = parser.add_subparsers(required=True, metavar="COMMAND")

    status = commands.add_parser("status", help="show backend, key status and presets")
    status.set_defaults(handler=cmd_status)

    presets = commands.add_parser("presets", help="list and validate presets")
    presets.set_defaults(handler=cmd_presets)

    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument(
        "--preset", required=True, help="preset name, or path to a preset file"
    )
    run_options.add_argument("--input", required=True, help="JSON array of {id, text, meta?}")
    run_options.add_argument(
        "--threshold", type=unit_interval, help="confidence cutoff (default: the preset's)"
    )
    run_options.add_argument(
        "--max-bytes",
        type=positive_int,
        default=DEFAULT_MAX_STATE_BYTES,
        help="items whose state is larger are not sent and go to review "
        f"(default: {DEFAULT_MAX_STATE_BYTES})",
    )
    run_options.add_argument(
        "--concurrency",
        type=positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"calls in flight at once (default: {DEFAULT_CONCURRENCY})",
    )
    run_options.add_argument(
        "--yes",
        action="store_true",
        help="send the items; without it jab only says what it would send, and where",
    )
    run_options.add_argument(
        "--debug", action="store_true", help="log what is sent, including item text"
    )

    ask = commands.add_parser(
        "ask", parents=[run_options], help="ask about every item and write one results file"
    )
    ask.add_argument("--out", required=True, help="results file (JSON lines)")
    ask.set_defaults(handler=cmd_ask)

    tri = commands.add_parser(
        "triage", parents=[run_options], help="sort items into confident, review and audit"
    )
    tri.add_argument("--out", required=True, help="directory for the output files")
    tri.add_argument(
        "--audit-rate",
        type=unit_interval,
        default=DEFAULT_AUDIT_RATE,
        help=f"share of confident items to sample for audit (default: {DEFAULT_AUDIT_RATE})",
    )
    tri.add_argument("--seed", type=int, help="seed for the audit sample")
    tri.set_defaults(handler=cmd_triage)

    return parser


def cmd_status(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    print(f"backend: {os.environ.get('JEV_BACKEND') or 'not set'}")
    for var in KEY_VARS.values():
        print(f"{var}: {'set' if os.environ.get(var) else 'not set'}")
    print(f"presets in {PRESETS_DIR}:")
    print_presets()
    return 0


def cmd_presets(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    return 0 if print_presets() else 1


def cmd_ask(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    preset, items = load_run_inputs(args)
    backend = backend_factory(os.environ)
    if not confirm_send(args, backend, preset, items):
        return NEEDS_CONFIRMATION
    result = run(
        backend,
        preset,
        items,
        threshold=args.threshold,
        max_state_bytes=args.max_bytes,
        concurrency=args.concurrency,
        debug=args.debug,
    )
    write_jsonl(
        Path(args.out), [r.to_json(with_text=r.bucket == NEEDS_REVIEW) for r in result.items]
    )
    print(json.dumps(result.summary(), indent=2))
    return 0


def cmd_triage(args: argparse.Namespace, backend_factory: BackendFactory) -> int:
    preset, items = load_run_inputs(args)
    backend = backend_factory(os.environ)
    if not confirm_send(args, backend, preset, items):
        return NEEDS_CONFIRMATION
    result = triage(
        backend,
        preset,
        items,
        threshold=args.threshold,
        audit_rate=args.audit_rate,
        seed=args.seed,
        max_state_bytes=args.max_bytes,
        concurrency=args.concurrency,
        debug=args.debug,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Confident lines stay short; the agent reads review and audit lines in full.
    write_jsonl(out / "confident.jsonl", [r.to_json() for r in result.confident])
    write_jsonl(
        out / "needs_review.jsonl", [r.to_json(with_text=True) for r in result.needs_review]
    )
    write_jsonl(out / "audit.jsonl", [r.to_json(with_text=True) for r in result.audit])
    summary = json.dumps(result.summary, indent=2)
    (out / "summary.json").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


def confirm_send(
    args: argparse.Namespace, backend: Backend, preset: Preset, items: list[dict]
) -> bool:
    """Say what will be sent and where, before the first call. Goes ahead only
    with --yes. The notice shows the template, never any item text."""
    if not items:
        return True
    where = f"{backend.destination} ({backend.name}, model {backend.model})"
    count = f"{len(items)} item{'' if len(items) == 1 else 's'}"
    if args.yes:
        print(f"jab: sending {count} to {where}", file=sys.stderr)
        return True
    template = json.dumps(preset.state_template, ensure_ascii=False)
    print(
        f"jab: this run would send {count} to {where}.\n"
        f"jab: each call carries one item, filled into this template from preset "
        f"{preset.name!r}, plus the preset's questions:\n"
        f"jab:   {template}\n"
        "jab: item ids, and meta fields the template does not use, are not sent.\n"
        "jab: nothing has been sent. Run again with --yes to send.",
        file=sys.stderr,
    )
    return False


def print_presets() -> bool:
    """Print each preset, or why it is invalid. Returns False if any is invalid."""
    paths = list_presets(PRESETS_DIR)
    if not paths:
        print("  (none)")
    all_valid = True
    for path in paths:
        try:
            preset = load_preset(path)
        except PresetError as exc:
            all_valid = False
            print(f"  {path.stem}: invalid: {exc}")
            continue
        print(f"  {preset.name}: {preset.description}")
    return all_valid


def load_run_inputs(args: argparse.Namespace) -> tuple[Preset, list[dict]]:
    preset = load_preset(find_preset(args.preset, PRESETS_DIR))
    try:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{args.input}: cannot read input: {exc}") from exc
    return preset, parse_items(data)


def write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return number

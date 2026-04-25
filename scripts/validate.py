#!/usr/bin/env python3
"""Validate open-plates procedure YAML files against the JSON Schema.

Usage:
    uv run python scripts/validate.py path/to/procedure.yaml [more.yaml ...]

Exits 0 if every file validates; exits 1 if any file fails validation or cannot
be loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "schema" / "procedure.schema.json"


def _format_pointer(absolute_path) -> str:
    """Render a jsonschema error path as a dotted/bracketed JSON pointer.

    e.g. ['minima', 3, 'visibility_sm'] -> 'minima[3].visibility_sm'
    """
    parts: list[str] = []
    for element in absolute_path:
        if isinstance(element, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{element}]"
            else:
                parts.append(f"[{element}]")
        else:
            parts.append(str(element))
    return ".".join(parts) if parts else "<root>"


def _load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_yaml(doc_path: Path) -> object:
    with doc_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _validate_one(
    doc_path: Path, validator: Draft202012Validator
) -> tuple[bool, list[str]]:
    """Return (ok, messages). Messages are one per error when not ok."""
    try:
        document = _load_yaml(doc_path)
    except (OSError, yaml.YAMLError) as exc:
        return False, [f"failed to load YAML: {exc}"]

    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    if not errors:
        return True, []
    formatted = [
        f"{_format_pointer(err.absolute_path)}: {err.message}" for err in errors
    ]
    return False, formatted


@click.command(help=__doc__)
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SCHEMA_PATH,
    show_default=True,
    help="Path to the JSON Schema file.",
)
def validate(paths: tuple[Path, ...], schema_path: Path) -> None:
    schema = _load_schema(schema_path)
    validator = Draft202012Validator(schema)

    failures = 0
    for path in paths:
        ok, messages = _validate_one(path, validator)
        if ok:
            click.echo(f"{path}: OK")
        else:
            failures += 1
            click.echo(f"{path}: FAIL", err=True)
            for msg in messages:
                click.echo(f"  {msg}", err=True)

    total = len(paths)
    passed = total - failures
    if total > 1:
        summary = f"{passed}/{total} files passed"
        click.echo(summary, err=(failures > 0))

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    validate()

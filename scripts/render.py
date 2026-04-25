#!/usr/bin/env python3
"""Render an open-plates procedure YAML to SVG, PDF, or PNG.

Usage:
    uv run python scripts/render.py examples/ugsb-ils-rwy-13.yaml out/ugsb-ils-rwy-13.pdf

Output format is chosen by the output file extension (.svg / .pdf / .png).
The input is validated against the repo's JSON Schema before rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from jsonschema import Draft202012Validator

from open_plates.legs import build_fix_table, compile_legs
from open_plates.render_svg import render_iap_svg

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "schema" / "procedure.schema.json"


def _format_pointer(absolute_path) -> str:
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


@click.command(help=__doc__)
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SCHEMA_PATH,
    show_default=True,
    help="Path to the JSON Schema file.",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip schema validation (for debugging).",
)
@click.option(
    "--scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Raster scale factor (PDF/PNG only). 2.0 = 2x pixels.",
)
def render(
    input_path: Path,
    output_path: Path,
    schema_path: Path,
    skip_validation: bool,
    scale: float,
) -> None:
    document = yaml.safe_load(input_path.read_text(encoding="utf-8"))

    if not skip_validation:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        errors = sorted(
            validator.iter_errors(document),
            key=lambda e: list(e.absolute_path),
        )
        if errors:
            click.echo(f"{input_path}: schema validation FAILED", err=True)
            for err in errors:
                click.echo(
                    f"  {_format_pointer(err.absolute_path)}: {err.message}",
                    err=True,
                )
            sys.exit(1)

    fixes = build_fix_table(document)
    primitives = compile_legs(document, fixes)
    svg = render_iap_svg(document, primitives)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".svg":
        output_path.write_text(svg, encoding="utf-8")
        size = len(svg)
    elif suffix in (".pdf", ".png"):
        import cairosvg

        convert = cairosvg.svg2pdf if suffix == ".pdf" else cairosvg.svg2png
        kwargs = {"scale": scale} if scale != 1.0 else {}
        data = convert(bytestring=svg.encode("utf-8"), **kwargs)
        output_path.write_bytes(data)
        size = len(data)
    else:
        click.echo(f"unsupported output extension: {suffix}", err=True)
        sys.exit(2)
    click.echo(f"wrote {output_path} ({size} bytes, {len(primitives)} primitives)")


if __name__ == "__main__":
    render()

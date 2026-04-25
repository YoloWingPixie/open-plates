"""Schema validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "procedure.schema.json"
UGSB_EXAMPLE_PATH = REPO_ROOT / "examples" / "ugsb-ils-rwy-13.yaml"
ETAD_EXAMPLE_PATH = REPO_ROOT / "examples" / "etad-ils-rwy-23-straight.yaml"
ETAD_ARC_EXAMPLE_PATH = REPO_ROOT / "examples" / "etad-vor-dme-rwy-05.yaml"
LSZH_EXAMPLE_PATH = REPO_ROOT / "examples" / "lszh-ils-z-rwy-14.yaml"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    return Draft202012Validator(schema)


def _errors(validator: Draft202012Validator, document: object) -> list[str]:
    return [
        f"{list(err.absolute_path)}: {err.message}"
        for err in validator.iter_errors(document)
    ]


def test_ugsb_example_validates(validator: Draft202012Validator) -> None:
    with UGSB_EXAMPLE_PATH.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    errors = _errors(validator, document)
    assert not errors, "UGSB example failed to validate:\n" + "\n".join(errors)


def test_etad_example_validates(validator: Draft202012Validator) -> None:
    with ETAD_EXAMPLE_PATH.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    errors = _errors(validator, document)
    assert not errors, "ETAD example failed to validate:\n" + "\n".join(errors)


def test_etad_arc_example_validates(validator: Draft202012Validator) -> None:
    """VOR/DME arc approach with an AF leg validates cleanly."""
    with ETAD_ARC_EXAMPLE_PATH.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    errors = _errors(validator, document)
    assert not errors, "ETAD arc example failed to validate:\n" + "\n".join(errors)


def test_lszh_example_validates(validator: Draft202012Validator) -> None:
    """Multi-IAF DME arc ILS into LSZH validates cleanly. Three approach
    transitions (AMIKI / RILAX / GIPOL), each ending in an AF leg to the
    shared intermediate fix INTER; exercises the schema for repeated AF
    terminators in separate approach_transition branches."""
    with LSZH_EXAMPLE_PATH.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    errors = _errors(validator, document)
    assert not errors, "LSZH example failed to validate:\n" + "\n".join(errors)


def _minimal_iap(extra_minima: list[dict]) -> dict:
    """A minimal IAP procedure document for exercising schema rules."""
    return {
        "airport_icao": "XTST",
        "procedure_id": "TEST APCH",
        "procedure_type": "iap",
        "approach_subtype": "ils",
        "amendment": "1",
        "airac_cycle": "2604",
        "effective_date": "2026-04-16",
        "mag_variation_deg": 0.0,
        "reference_datum": "WGS84",
        "tdze_ft_msl": 0,
        "airport_elev_ft_msl": 0,
        "fixes": [
            {
                "id": "RW01",
                "type": "runway_threshold",
                "name": "RWY 01 THRESHOLD",
                "lat": 0.0,
                "lon": 0.0,
                "elevation_ft": 0,
            },
        ],
        "missed_approach": {
            "text_description": "Climb straight ahead to 2000.",
            "legs": [
                {
                    "terminator": "CA",
                    "transition_type": "missed_approach",
                    "from_fix_id": "RW01",
                    "course_deg": 10,
                    "altitude_constraint": {
                        "altitude_description": "at_or_above",
                        "altitude_1_ft": 2000,
                    },
                },
            ],
        },
        "minima": extra_minima,
    }


def test_not_authorized_minima_row_validates(validator: Draft202012Validator) -> None:
    """An NA minima row without visibility_sm must validate post-fix."""
    document = _minimal_iap(
        extra_minima=[
            {
                "variant": "circling",
                "category": "D",
                "not_authorized": True,
            },
            {
                "variant": "s_ils",
                "category": "A",
                "da_ft_msl": 232,
                "dh_ft_agl": 200,
                "visibility_sm": 0.50,
                "rvr_ft": 2400,
            },
        ],
    )
    errors = _errors(validator, document)
    assert not errors, "NA minima row should validate:\n" + "\n".join(errors)


def test_authorized_minima_row_still_requires_visibility(
    validator: Draft202012Validator,
) -> None:
    """Authorized rows (no not_authorized flag) must still require visibility_sm."""
    document = _minimal_iap(
        extra_minima=[
            {
                "variant": "s_ils",
                "category": "A",
                "da_ft_msl": 232,
                "dh_ft_agl": 200,
                # visibility_sm deliberately omitted
            },
        ],
    )
    errors = _errors(validator, document)
    assert any("visibility_sm" in e for e in errors), (
        "Authorized minima row without visibility_sm should fail validation, got: "
        + ("no errors" if not errors else "\n".join(errors))
    )


def test_not_authorized_false_still_requires_visibility(
    validator: Draft202012Validator,
) -> None:
    """not_authorized: false behaves identically to omitting the flag."""
    document = _minimal_iap(
        extra_minima=[
            {
                "variant": "s_ils",
                "category": "A",
                "not_authorized": False,
                "da_ft_msl": 232,
                "dh_ft_agl": 200,
                # visibility_sm deliberately omitted
            },
        ],
    )
    errors = _errors(validator, document)
    assert any("visibility_sm" in e for e in errors), (
        "not_authorized: false without visibility_sm should still fail: "
        + ("no errors" if not errors else "\n".join(errors))
    )

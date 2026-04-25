"""Tests for the tangent-arc constructor (planar approximation).

Post-refactor (April 2026): these cover just the two primitives still
used by the codebase — :func:`tangent_arc` and
:func:`flyby_fillet_line_to_line`. The old Dubins-CS + line-to-arc /
arc-to-line fillet machinery has been removed along with the
physics-based turn-radius model; see :mod:`open_plates.trajectory`.
"""

from __future__ import annotations

import pytest

from open_plates.geodesy import (
    destination,
    great_circle_distance_nm,
)
from open_plates.tangent_arc import (
    flyby_fillet_line_to_line,
    tangent_arc,
)


# ---------------------------------------------------------------------------
# tangent_arc
# ---------------------------------------------------------------------------


def test_right_90_at_origin() -> None:
    """Inbound 000 (north), outbound 090 (east), r = 1 NM. Turn right."""
    origin = (0.0, 0.0)
    arc_start, arc_end, arc = tangent_arc(
        inbound_course_deg=0.0,
        outbound_course_deg=90.0,
        turn_fix=origin,
        turn_radius_nm=1.0,
    )

    # Anticipation distance d = r * tan(45°) = 1 NM.
    assert great_circle_distance_nm(arc_start, origin) == pytest.approx(1.0, abs=1e-6)
    assert great_circle_distance_nm(arc_end, origin) == pytest.approx(1.0, abs=1e-6)
    assert arc_start[0] < 0
    assert arc_start[1] == pytest.approx(0.0, abs=1e-6)
    assert arc_end[0] == pytest.approx(0.0, abs=1e-6)
    assert arc_end[1] > 0
    assert arc.radius_nm == pytest.approx(1.0)
    assert arc.clockwise is True


def test_left_90_mirrors_right() -> None:
    origin = (0.0, 0.0)
    arc_start, arc_end, arc = tangent_arc(
        inbound_course_deg=0.0,
        outbound_course_deg=270.0,
        turn_fix=origin,
        turn_radius_nm=1.0,
    )

    assert great_circle_distance_nm(arc_start, origin) == pytest.approx(1.0, abs=1e-6)
    assert great_circle_distance_nm(arc_end, origin) == pytest.approx(1.0, abs=1e-6)
    assert arc_start[0] < 0
    assert arc_end[1] < 0
    assert arc.clockwise is False
    assert arc.radius_nm == pytest.approx(1.0)


def test_straight_through_returns_zero_radius_arc() -> None:
    origin = (41.0, 41.0)
    arc_start, arc_end, arc = tangent_arc(
        inbound_course_deg=90.0,
        outbound_course_deg=90.3,
        turn_fix=origin,
        turn_radius_nm=1.0,
    )
    assert arc_start == origin
    assert arc_end == origin
    assert arc.radius_nm == 0.0
    assert arc.center == origin


def test_anticipation_matches_formula() -> None:
    """d = r * tan(Δ/2) at 60° turn → d = r * tan(30°) ≈ 0.5774 NM for r=1."""
    from math import radians, tan

    origin = (30.0, 30.0)
    r = 1.0
    arc_start, arc_end, _ = tangent_arc(
        inbound_course_deg=0.0,
        outbound_course_deg=60.0,
        turn_fix=origin,
        turn_radius_nm=r,
    )
    expected = r * tan(radians(30.0))
    assert great_circle_distance_nm(arc_start, origin) == pytest.approx(
        expected, abs=1e-4
    )
    assert great_circle_distance_nm(arc_end, origin) == pytest.approx(
        expected, abs=1e-4
    )


# ---------------------------------------------------------------------------
# flyby_fillet_line_to_line — the one fillet the trajectory builder uses.
# ---------------------------------------------------------------------------


def test_flyby_line_to_line_90_deg_turn() -> None:
    """Inbound east (090°), outbound north (000°), fix at origin, R = 1 NM."""
    fix = (0.0, 0.0)
    prev_start = destination(fix, 270.0, 5.0)
    result = flyby_fillet_line_to_line(
        prev_start=prev_start,
        prev_end=fix,
        fix=fix,
        outbound_course_deg=0.0,
        turn_radius_nm=1.0,
    )
    assert result is not None
    tangent_in, arc, tangent_out = result

    assert great_circle_distance_nm(tangent_in, fix) == pytest.approx(1.0, abs=1e-4)
    assert great_circle_distance_nm(tangent_out, fix) == pytest.approx(1.0, abs=1e-4)
    assert tangent_in[0] == pytest.approx(0.0, abs=1e-4)
    assert tangent_in[1] < 0
    assert tangent_out[0] > 0
    assert tangent_out[1] == pytest.approx(0.0, abs=1e-4)
    assert arc.clockwise is False
    assert arc.radius_nm == pytest.approx(1.0)
    assert great_circle_distance_nm(arc.center, tangent_in) == pytest.approx(
        1.0, abs=1e-4
    )
    assert great_circle_distance_nm(arc.center, tangent_out) == pytest.approx(
        1.0, abs=1e-4
    )


def test_flyby_line_to_line_straight_through_returns_none() -> None:
    fix = (10.0, 10.0)
    prev_start = destination(fix, 270.0, 5.0)
    result = flyby_fillet_line_to_line(
        prev_start=prev_start,
        prev_end=fix,
        fix=fix,
        outbound_course_deg=90.0,
        turn_radius_nm=1.0,
    )
    assert result is None


def test_flyby_line_to_line_small_cosmetic_radius() -> None:
    """Trajectory builder passes a tiny 0.15 NM bend radius. Verify the
    fillet scales linearly — setback matches R / tan(45°) = 0.15 NM for
    a 90° turn and the emitted arc has radius 0.15 NM."""
    fix = (40.0, -90.0)
    prev_start = destination(fix, 270.0, 2.0)
    result = flyby_fillet_line_to_line(
        prev_start=prev_start,
        prev_end=fix,
        fix=fix,
        outbound_course_deg=0.0,
        turn_radius_nm=0.15,
    )
    assert result is not None
    tangent_in, arc, tangent_out = result
    assert arc.radius_nm == pytest.approx(0.15, rel=1e-6)
    # Planar approximation at 40°N on a 0.15 NM scale is accurate to
    # well under a pixel; allow 0.001 NM of slack.
    assert great_circle_distance_nm(tangent_in, fix) == pytest.approx(0.15, abs=1e-3)
    assert great_circle_distance_nm(tangent_out, fix) == pytest.approx(
        0.15, abs=1e-3
    )

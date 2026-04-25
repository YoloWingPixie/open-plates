"""Geodesy primitive tests."""

from __future__ import annotations

import pytest

from open_plates.geodesy import (
    EARTH_RADIUS_NM,
    destination,
    great_circle_distance_nm,
    initial_bearing_deg,
    intersection,
)

KJFK = (40.6413, -73.7781)
EGLL = (51.4700, -0.4543)
KLAX = (33.9416, -118.4085)


def test_distance_jfk_lhr() -> None:
    assert great_circle_distance_nm(KJFK, EGLL) == pytest.approx(2998, abs=10)


def test_bearing_jfk_lhr() -> None:
    assert initial_bearing_deg(KJFK, EGLL) == pytest.approx(51, abs=1)


def test_distance_lax_jfk() -> None:
    assert great_circle_distance_nm(KLAX, KJFK) == pytest.approx(2144, abs=5)


def test_bearing_lax_jfk() -> None:
    assert initial_bearing_deg(KLAX, KJFK) == pytest.approx(66, abs=1)


def test_destination_equator_east() -> None:
    lat, lon = destination((0.0, 0.0), 90.0, 60.0)
    assert lat == pytest.approx(0.0, abs=1e-6)
    assert lon == pytest.approx(1.0, abs=0.01)


def test_destination_north_pole() -> None:
    distance = 90.0 * (2 * 3.141592653589793 * EARTH_RADIUS_NM) / 360.0
    lat, _ = destination((0.0, 0.0), 0.0, distance)
    assert lat == pytest.approx(90.0, abs=1e-3)


def test_round_trip() -> None:
    start = KJFK
    end = destination(start, 80.0, 1500.0)
    assert great_circle_distance_nm(start, end) == pytest.approx(1500.0, rel=1e-6)


def test_antimeridian_crossing() -> None:
    d = great_circle_distance_nm((0.0, 179.0), (0.0, -179.0))
    assert d == pytest.approx(120.0, abs=1.0)


def test_bearing_range() -> None:
    b = initial_bearing_deg((0.0, 0.0), (0.0, -1.0))
    assert 0.0 <= b < 360.0
    assert b == pytest.approx(270.0, abs=1e-6)


def test_intersection_known_fix() -> None:
    p1 = (51.8853, 0.2545)
    p2 = (49.0034, 2.5735)
    fix = intersection(p1, 108.547, p2, 32.435)
    assert fix is not None
    assert fix[0] == pytest.approx(50.9078, abs=0.01)
    assert fix[1] == pytest.approx(4.5084, abs=0.01)


def test_intersection_coincident_point_returns_none() -> None:
    """Same point, any bearings — ambiguous."""
    p = (10.0, 20.0)
    assert intersection(p, 45.0, p, 90.0) is None


def test_intersection_roundtrip_via_destination() -> None:
    a = (40.0, -75.0)
    b = (42.0, -70.0)
    meet = destination(a, 60.0, 100.0)
    brng_a = initial_bearing_deg(a, meet)
    brng_b = initial_bearing_deg(b, meet)
    fix = intersection(a, brng_a, b, brng_b)
    assert fix is not None
    assert fix[0] == pytest.approx(meet[0], abs=1e-3)
    assert fix[1] == pytest.approx(meet[1], abs=1e-3)

"""Spherical-Earth geodesy primitives (Chris Veness formulas, MIT).

Source: https://www.movable-type.co.uk/scripts/latlong.html
Spherical approximation; not for actual flight ops.
"""

from __future__ import annotations

from math import asin, atan2, cos, degrees, pi, radians, sin, sqrt

LatLon = tuple[float, float]

EARTH_RADIUS_NM = 3440.065


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi] for safe acos/asin."""
    return lo if x < lo else hi if x > hi else x


def _norm_deg(deg: float) -> float:
    """Normalize degrees to [0, 360)."""
    return deg % 360.0


def great_circle_distance_nm(a: LatLon, b: LatLon) -> float:
    """Haversine great-circle distance in NM. Inputs: (lat, lon) degrees."""
    lat1, lon1 = radians(a[0]), radians(a[1])
    lat2, lon2 = radians(b[0]), radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * asin(sqrt(_clamp(h, 0.0, 1.0)))


def initial_bearing_deg(a: LatLon, b: LatLon) -> float:
    """Forward azimuth at `a` toward `b`, degrees true in [0, 360)."""
    lat1, lat2 = radians(a[0]), radians(b[0])
    dlon = radians(b[1] - a[1])
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return _norm_deg(degrees(atan2(y, x)))


def destination(start: LatLon, bearing_deg: float, distance_nm: float) -> LatLon:
    """Point reached flying `bearing_deg` for `distance_nm` from `start`."""
    ang = distance_nm / EARTH_RADIUS_NM
    brng = radians(bearing_deg)
    lat1, lon1 = radians(start[0]), radians(start[1])
    sin_lat2 = sin(lat1) * cos(ang) + cos(lat1) * sin(ang) * cos(brng)
    lat2 = asin(_clamp(sin_lat2))
    y = sin(brng) * sin(ang) * cos(lat1)
    x = cos(ang) - sin(lat1) * sin(lat2)
    lon2 = lon1 + atan2(y, x)
    lon2_deg = ((degrees(lon2) + 540.0) % 360.0) - 180.0
    return (degrees(lat2), lon2_deg)


def intersection(
    p1: LatLon, brng1_deg: float, p2: LatLon, brng2_deg: float
) -> LatLon | None:
    """Intersection of two great-circle radials; None if parallel/ambiguous/antipodal."""
    lat1, lon1 = radians(p1[0]), radians(p1[1])
    lat2, lon2 = radians(p2[0]), radians(p2[1])
    brng13, brng23 = radians(brng1_deg), radians(brng2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    d12 = 2 * asin(
        sqrt(
            _clamp(
                sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2,
                0.0,
                1.0,
            )
        )
    )
    if d12 == 0:
        return None

    cos_theta_a = (sin(lat2) - sin(lat1) * cos(d12)) / (sin(d12) * cos(lat1))
    cos_theta_b = (sin(lat1) - sin(lat2) * cos(d12)) / (sin(d12) * cos(lat2))
    theta_a = atan2(sqrt(_clamp(1 - cos_theta_a * cos_theta_a, 0.0, 1.0)), cos_theta_a)
    theta_b = atan2(sqrt(_clamp(1 - cos_theta_b * cos_theta_b, 0.0, 1.0)), cos_theta_b)

    if sin(lon2 - lon1) > 0:
        theta12 = theta_a
        theta21 = 2 * pi - theta_b
    else:
        theta12 = 2 * pi - theta_a
        theta21 = theta_b

    alpha1 = brng13 - theta12
    alpha2 = theta21 - brng23

    if sin(alpha1) == 0 and sin(alpha2) == 0:
        return None
    if sin(alpha1) * sin(alpha2) < 0:
        return None

    cos_alpha3 = -cos(alpha1) * cos(alpha2) + sin(alpha1) * sin(alpha2) * cos(d12)
    d13 = atan2(
        sin(d12) * sin(alpha1) * sin(alpha2),
        cos(alpha2) + cos(alpha1) * cos_alpha3,
    )
    lat3 = asin(_clamp(sin(lat1) * cos(d13) + cos(lat1) * sin(d13) * cos(brng13)))
    dlon13 = atan2(
        sin(brng13) * sin(d13) * cos(lat1),
        cos(d13) - sin(lat1) * sin(lat3),
    )
    lon3 = lon1 + dlon13
    lon3_deg = ((degrees(lon3) + 540.0) % 360.0) - 180.0
    return (degrees(lat3), lon3_deg)

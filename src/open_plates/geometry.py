"""Drawable geometry primitives emitted by the leg dispatcher.

Pure data only — no behavior. Coordinates are WGS-84 decimal degrees
(lat, lon) tuples. Arc angles are bearings in degrees true (measured from
the arc center, 0 = north, 90 = east). Distances are nautical miles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .geodesy import LatLon

TurnDirection = Literal["left", "right"]


@dataclass(frozen=True)
class Segment:
    """Straight great-circle segment between two points."""

    start: LatLon
    end: LatLon


@dataclass(frozen=True)
class Arc:
    """Circular arc in the local tangent plane near `center`.

    Bearings are from the arc center to the respective endpoint, degrees
    true. `clockwise` indicates the traversal direction from start to end.
    """

    center: LatLon
    radius_nm: float
    start_bearing_deg: float
    end_bearing_deg: float
    clockwise: bool


@dataclass(frozen=True)
class Hold:
    """Racetrack hold. Rendered symbolically by the chart layer."""

    fix: LatLon
    inbound_course_deg: float
    turn_direction: TurnDirection
    leg_length_nm: float | None = None
    leg_time_sec: float | None = None


Primitive = Segment | Arc | Hold


# One continuous flyable trajectory per state of flight (entry fix →
# termination fix). The renderer emits ONE casing + ONE core polyline per
# state so adjacent-primitive casing overlap and sub-pixel endpoint drift vanish.
FlightPhase = Literal["approach", "missed"]


@dataclass
class FlightState:
    """One continuous flyable trajectory.

    ``primitives`` is the underlying Segment/Arc/Hold list (unchanged
    from the compiler's output, so existing consumers — bbox masks,
    tests on primitive shape — keep working).

    ``points`` is the flattened polyline: arcs are sampled, segment
    endpoints are deduped against their predecessor, Holds are omitted
    (rendered separately by the renderer as a racetrack symbol).

    ``holds`` carries any Hold primitives that belong to this state so
    the renderer can emit them alongside the polyline without needing
    to scan the flat primitive list again.
    """

    phase: FlightPhase
    label: str
    primitives: list[Primitive] = field(default_factory=list)
    points: list[LatLon] = field(default_factory=list)
    holds: list[Hold] = field(default_factory=list)

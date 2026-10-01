"""NBA court geometry in feet. Origin = one corner of the court; x runs along the length
(0..94), y along the width (0..50). This matches the raw SportVU frame.

Left basket is at (5.25, 25); right basket at (88.75, 25). Values from the NBA rule book.
"""

from __future__ import annotations

import numpy as np

LENGTH = 94.0
WIDTH = 50.0
HALF = LENGTH / 2  # 47.0

HOOP_X_LEFT = 5.25  # backboard face at 4 ft, hoop centre 15 in (1.25 ft) in front of it
HOOP_X_RIGHT = LENGTH - HOOP_X_LEFT  # 88.75
HOOP_Y = WIDTH / 2  # 25.0
HOOP_RADIUS = 0.75  # 18 in diameter
BACKBOARD_X_LEFT = 4.0
BACKBOARD_HALF_WIDTH = 3.0  # 6 ft wide

THREE_PT_RADIUS = 23.75  # from hoop centre
THREE_PT_CORNER_Y_OFFSET = 22.0  # corner three: 22 ft from hoop centre laterally
THREE_PT_CORNER_LENGTH = 14.0  # straight segment length from baseline

LANE_WIDTH = 16.0  # outer paint ("key") width
LANE_LENGTH = 19.0  # baseline to free-throw line
FT_CIRCLE_RADIUS = 6.0
RESTRICTED_RADIUS = 4.0  # from hoop centre
CENTER_CIRCLE_RADIUS = 6.0

HOOP_LEFT = np.array([HOOP_X_LEFT, HOOP_Y])
HOOP_RIGHT = np.array([HOOP_X_RIGHT, HOOP_Y])


def hoop_for_side(left: bool) -> np.ndarray:
    return HOOP_LEFT if left else HOOP_RIGHT


def three_point_line(left: bool = True, n: int = 200) -> np.ndarray:
    """(n + 2, 2) polyline of the three-point line for the left (or right) basket."""
    hx = HOOP_X_LEFT if left else HOOP_X_RIGHT
    sign = 1.0 if left else -1.0
    # the arc meets the corner straight lines where |y - HOOP_Y| == 22 ft
    theta_max = np.arcsin(THREE_PT_CORNER_Y_OFFSET / THREE_PT_RADIUS)
    th = np.linspace(-theta_max, theta_max, n)
    arc_x = hx + sign * THREE_PT_RADIUS * np.cos(th)
    arc_y = HOOP_Y + THREE_PT_RADIUS * np.sin(th)
    x0 = 0.0 if left else LENGTH
    top = np.array([[x0, HOOP_Y + THREE_PT_CORNER_Y_OFFSET]])
    bot = np.array([[x0, HOOP_Y - THREE_PT_CORNER_Y_OFFSET]])
    return np.vstack([bot, np.column_stack([arc_x, arc_y]), top])


def is_three_point_location(x: np.ndarray, y: np.ndarray, left: bool = True) -> np.ndarray:
    """Vectorised: True where a shot from (x, y) is beyond the three-point line."""
    hoop = hoop_for_side(left)
    dx = np.asarray(x) - hoop[0]
    dy = np.asarray(y) - hoop[1]
    dist = np.hypot(dx, dy)
    x_from_baseline = np.asarray(x) if left else LENGTH - np.asarray(x)
    corner = (x_from_baseline <= THREE_PT_CORNER_LENGTH) & (np.abs(dy) >= THREE_PT_CORNER_Y_OFFSET)
    return corner | (dist >= THREE_PT_RADIUS)


def to_left_half(x: np.ndarray, y: np.ndarray, attacking_left) -> tuple[np.ndarray, np.ndarray]:
    """Reflect coordinates so that the offence always attacks the left basket.

    ``attacking_left`` is broadcastable to x/y: True where the offence already attacks the
    left basket (no change), False where it attacks the right basket (rotate 180 deg about
    the court centre: both x and y are flipped, which preserves handedness / strong side).
    """
    flip = ~np.asarray(attacking_left, dtype=bool)
    xo = np.where(flip, LENGTH - np.asarray(x), x)
    yo = np.where(flip, WIDTH - np.asarray(y), y)
    return xo, yo

"""Matplotlib court drawing (full court, feet, same frame as the raw data)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle

from nbacore import court as C


def draw_court(ax: Axes | None = None, color: str = "black", lw: float = 1.0) -> Axes:
    """Draw a full NBA court on ``ax`` (created if None) and return the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9.4, 5.0))
    kw = dict(color=color, lw=lw, fill=False)

    # boundary + half court
    ax.add_patch(Rectangle((0, 0), C.LENGTH, C.WIDTH, **kw))
    ax.plot([C.HALF, C.HALF], [0, C.WIDTH], color=color, lw=lw)
    ax.add_patch(Circle((C.HALF, C.HOOP_Y), C.CENTER_CIRCLE_RADIUS, **kw))

    for left in (True, False):
        sign = 1 if left else -1
        base_x = 0.0 if left else C.LENGTH
        hoop_x = C.HOOP_X_LEFT if left else C.HOOP_X_RIGHT
        # lane (outer key)
        ax.add_patch(
            Rectangle(
                (base_x if left else base_x - C.LANE_LENGTH, C.HOOP_Y - C.LANE_WIDTH / 2),
                C.LANE_LENGTH,
                C.LANE_WIDTH,
                **kw,
            )
        )
        # free-throw circle (solid half towards the court, dashed half inside the lane)
        ft_x = base_x + sign * C.LANE_LENGTH
        ax.add_patch(
            Arc(
                (ft_x, C.HOOP_Y),
                2 * C.FT_CIRCLE_RADIUS,
                2 * C.FT_CIRCLE_RADIUS,
                theta1=-90 if left else 90,
                theta2=90 if left else 270,
                color=color,
                lw=lw,
            )
        )
        ax.add_patch(
            Arc(
                (ft_x, C.HOOP_Y),
                2 * C.FT_CIRCLE_RADIUS,
                2 * C.FT_CIRCLE_RADIUS,
                theta1=90 if left else -90,
                theta2=270 if left else 90,
                color=color,
                lw=lw,
                linestyle="--",
            )
        )
        # backboard + hoop + restricted area
        bb_x = base_x + sign * C.BACKBOARD_X_LEFT
        ax.plot(
            [bb_x, bb_x],
            [C.HOOP_Y - C.BACKBOARD_HALF_WIDTH, C.HOOP_Y + C.BACKBOARD_HALF_WIDTH],
            color=color,
            lw=lw * 2,
        )
        ax.add_patch(Circle((hoop_x, C.HOOP_Y), C.HOOP_RADIUS, **kw))
        ax.add_patch(
            Arc(
                (hoop_x, C.HOOP_Y),
                2 * C.RESTRICTED_RADIUS,
                2 * C.RESTRICTED_RADIUS,
                theta1=-90 if left else 90,
                theta2=90 if left else 270,
                color=color,
                lw=lw,
            )
        )
        # three-point line
        tp = C.three_point_line(left=left)
        ax.plot(tp[:, 0], tp[:, 1], color=color, lw=lw)

    ax.set_xlim(-2, C.LENGTH + 2)
    ax.set_ylim(-2, C.WIDTH + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax

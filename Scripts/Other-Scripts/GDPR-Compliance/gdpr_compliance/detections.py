"""The Detection value type shared by every detector and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Detection:
    """One sensitive region, in pixel coordinates of the analysed image.

    ``pad_fraction`` is the safety margin the owning detector wants applied
    around this region before redaction; the pipeline calls :meth:`padded`
    and reports both the raw and the padded box.
    """

    x: int
    y: int
    w: int
    h: int
    label: str
    score: float
    detector: str
    pad_fraction: float = 0.15

    def padded(self, width: int, height: int) -> "Detection":
        """Return a copy grown by ``pad_fraction`` per side, clipped to bounds."""
        pad_x = int(round(self.w * self.pad_fraction))
        pad_y = int(round(self.h * self.pad_fraction))
        x0 = max(0, self.x - pad_x)
        y0 = max(0, self.y - pad_y)
        x1 = min(width, self.x + self.w + pad_x)
        y1 = min(height, self.y + self.h + pad_y)
        return replace(self, x=x0, y=y0, w=x1 - x0, h=y1 - y0, pad_fraction=0.0)

    def box_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

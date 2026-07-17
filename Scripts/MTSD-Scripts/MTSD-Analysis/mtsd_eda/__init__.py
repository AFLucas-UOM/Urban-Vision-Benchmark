"""Support package for the MTSD exploratory data analysis.

Modules
-------
config       Shared filesystem paths and scan settings.
inventory    Image discovery + EXIF metadata extraction (cached to CSV).
coco         Loading and validation of the QA-verified COCO annotation files.
annotations  Annotation-level analysis (summaries, distributions, co-occurrence,
             bounding boxes, locality aggregation).
geo          Geodesic helpers and Malta/Gozo locality assignment.
plotstyle    Publication-quality matplotlib styling and figure export.
"""

from . import annotations, config, coco, geo, inventory, plotstyle

__all__ = ["annotations", "config", "coco", "geo", "inventory", "plotstyle"]

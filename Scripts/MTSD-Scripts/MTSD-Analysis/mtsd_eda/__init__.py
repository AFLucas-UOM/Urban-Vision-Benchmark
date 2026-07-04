"""Support package for the MTSD exploratory data analysis.

Modules
-------
config     Shared filesystem paths and scan settings.
inventory  Image discovery + EXIF metadata extraction (cached to CSV).
coco       Loading and validation of the QA-verified COCO annotation files.
geo        Geodesic helpers and Malta/Gozo locality assignment.
plotstyle  Publication-quality matplotlib styling and figure export.
"""

from . import config, coco, geo, inventory, plotstyle

__all__ = ["config", "coco", "geo", "inventory", "plotstyle"]

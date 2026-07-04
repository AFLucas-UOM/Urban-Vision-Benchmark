"""MTSD GDPR Compliance - privacy redaction for dataset images.

Detects privacy-sensitive regions (human faces, vehicle licence plates,
QR codes) and redacts them with size-scaled Gaussian blur. Previews are
written to a separate output tree; originals are only ever replaced by the
explicit, confirmed ``apply`` command.

Package layout (one concern per module):

- ``detections``  - the :class:`Detection` value type (raw + padded boxes)
- ``detectors``   - detector interface, implementations, registry, backends
- ``imaging``     - image discovery and EXIF-correct loading/saving
- ``blur``        - redaction primitives (Gaussian / pixelation) and sheets
- ``reporting``   - detection_report.json / detection_summary.csv writers
- ``pipeline``    - preview and apply orchestration
"""

__version__ = "2.0.0"
TOOL_NAME = f"MTSD GDPR Compliance v{__version__}"

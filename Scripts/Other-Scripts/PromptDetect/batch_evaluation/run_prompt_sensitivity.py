#!/usr/bin/env python3
"""Convenience wrapper: the dissertation runner with the prompt-sensitivity protocol.

This adds no evaluation logic of its own — it delegates to
``run_dissertation_protocol.main`` and merely injects
``--protocol prompt_protocols/prompt_sensitivity_protocol.yaml`` when the
caller did not pick a protocol explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

from run_dissertation_protocol import SENSITIVITY_PROTOCOL, main


def wrapper_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--protocol" not in argv:
        argv = ["--protocol", str(SENSITIVITY_PROTOCOL), *argv]
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(wrapper_main())

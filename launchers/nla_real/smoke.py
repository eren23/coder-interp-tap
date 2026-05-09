"""Smoke variant — same pipeline as pilot, gated by env (NUM_PROMPTS=3,
POSITIONS_PER_PROMPT=1 in the project YAML's smoke variant block).

Crucible runs us as `python3 -u launchers/nla_real/smoke.py` so the
package import path "launchers.nla_real.pilot" is NOT on sys.path. Add
this script's parent dir to sys.path so `import pilot` works.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pilot import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

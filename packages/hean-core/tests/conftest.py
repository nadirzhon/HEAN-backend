"""Conftest for hean-core package tests.

Inserts all backend workspace package source directories into sys.path before
any test module is imported.  This is necessary because the project uses a uv
workspace where packages are not installed into the virtual environment — they
are provided via PYTHONPATH when running via 'make test'.  When running the
package's own test suite directly (e.g. 'pytest packages/hean-core/tests/')
we must replicate that PYTHONPATH ourselves.
"""

import sys
from pathlib import Path

# Backend workspace root (two levels up from this conftest)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_PACKAGES = [
    "hean-core",
    "hean-exchange",
    "hean-portfolio",
    "hean-risk",
    "hean-execution",
    "hean-strategies",
    "hean-physics",
    "hean-intelligence",
    "hean-observability",
    "hean-symbiont",
    "hean-api",
    "hean-app",
]

for _pkg in _PACKAGES:
    _src = str(_BACKEND_ROOT / "packages" / _pkg / "src")
    if _src not in sys.path:
        # Insert at the front so backend packages shadow any system installs
        sys.path.insert(0, _src)

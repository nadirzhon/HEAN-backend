"""HEAN Live Testnet Trading — Docker entrypoint.

Thin wrapper that starts TradingSystem in 'run' mode.
Equivalent to: python -m hean.main run
"""

import sys

sys.argv = [sys.argv[0], "run"]

from hean.main import main  # noqa: E402

if __name__ == "__main__":
    main()

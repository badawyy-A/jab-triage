"""Run jab straight from the skill folder, without installing it.

python scripts/jab.py status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jab_triage.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

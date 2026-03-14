from __future__ import annotations

import sys
from pathlib import Path


# Ensure the Streamlit service modules are importable when tests are executed
# from the container workdir.
STREAMLIT_ROOT = Path(__file__).resolve().parents[1]
if str(STREAMLIT_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_ROOT))

"""
Centralized dataset path definitions.

Defaults match the local layered dataset already present on disk. Override via
environment variables when needed (e.g., for remote runs).
"""
from pathlib import Path
import os


# Root containing per-sample rendered RGBA assets (background + components).
RENDERED_ROOT = Path(
    os.getenv(
        "QIL_RENDERED_ROOT",
        "/home/ubuntu/jjseol/layer_data/inpainting_250k_subset_rendered",
    )
)

# Root containing per-sample JSON layout/config metadata.
JSON_ROOT = Path(
    os.getenv(
        "QIL_JSON_ROOT",
        "/home/ubuntu/jjseol/layer_data/inpainting_250k_subset",
    )
)

# Optional explicit composite path root; if absent, composite will be computed
# from background + components.
COMPOSITE_ROOT = Path(os.getenv("QIL_COMPOSITE_ROOT", ""))




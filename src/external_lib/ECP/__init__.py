from __future__ import annotations

from pathlib import Path

from external_lib._vendor import configure_vendored_package


_UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "external_lib" / "ECP"

# Expose the vendored ECP source tree as the package path while also
# redirecting ECP's absolute intra-repository imports (for example `utils.*`).
__path__ = configure_vendored_package(__name__, _UPSTREAM_ROOT)


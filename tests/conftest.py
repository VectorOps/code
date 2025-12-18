from __future__ import annotations

from vocode import diagnostics as _diagnostics


def pytest_configure(config) -> None:
    # Reuse the central warning/filter setup for the test process.
    _diagnostics.setup_fault_handlers()

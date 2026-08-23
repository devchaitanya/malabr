"""Server components for the MALABR runtime.

Deliberately imports NOTHING at package import time. The previous version did
`from .runtime import run_server`, which dragged every importer through
runtime -> ML.Request -> flatbuffers. That made `engine.py` impossible to
import on its own, defeating section 7's requirement that the SlotAllocator be
testable standalone, and it failed outright wherever flatbuffers is absent.

Import the submodule you need directly.
"""

__all__ = ["config", "protocol", "engine", "runtime"]

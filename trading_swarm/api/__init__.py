"""REST API (FastAPI) for the trading swarm.

``create_app`` and the uvicorn entry point ``app`` live in :mod:`trading_swarm.api.main`;
they are exposed here lazily so that importing a submodule (``schemas``, ``security``)
does not build the module-level application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

    from trading_swarm.api.main import create_app

    app: FastAPI

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from trading_swarm.api import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

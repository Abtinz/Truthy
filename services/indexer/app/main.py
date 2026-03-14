from __future__ import annotations

from fastapi import Depends
from fastapi import FastAPI

from app.api.routes import router
from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.dependencies import get_policy_freshness_cache

app = FastAPI(title="Truthy Indexer")
app.include_router(router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return the root status payload for the indexer service.

    Args:
        None.

    Returns:
        dict[str, str]: Minimal service status payload.
    """

    return {"service": "truthy-indexer", "status": "ok"}


@app.get("/health")
def read_health(
    policy_cache: PolicyFreshnessCache = Depends(get_policy_freshness_cache),
) -> dict[str, str]:
    """Return a health payload after validating Redis connectivity.

    The route performs a lightweight Redis operation through the cache client so
    container orchestration can confirm the freshness-cache dependency is
    reachable.

    Args:
        policy_cache: Redis freshness cache dependency.

    Returns:
        dict[str, str]: Health payload for the indexer service.
    """

    policy_cache.list_entries()
    return {"service": "truthy-indexer", "status": "ok"}

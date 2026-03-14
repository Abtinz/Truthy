from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.dependencies import get_policy_freshness_cache

router = APIRouter()


@router.get("/cache/policy-freshness")
def read_policy_freshness_cache(
    policy_cache: PolicyFreshnessCache = Depends(get_policy_freshness_cache),
) -> dict[str, object]:
    """Return the Redis-tracked policy freshness records.

    This endpoint acts as a lightweight log view for the Redis optimization
    layer. It exposes only the minimal cache contents requested by the product
    design: the tracked source URL and its latest stored `Date modified` value.

    Args:
        policy_cache: Redis freshness cache dependency.

    Returns:
        dict[str, object]: Structured cache-log payload for operator review.
    """

    entries = policy_cache.list_entries()
    return {
        "cache_name": "policy_freshness",
        "entry_count": len(entries),
        "entries": [
            {
                "source_url": entry.source_url,
                "modified_date": entry.modified_date,
            }
            for entry in entries
        ],
    }

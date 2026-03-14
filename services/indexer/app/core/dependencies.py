from __future__ import annotations

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.config import IndexerSettings


def get_policy_freshness_cache() -> PolicyFreshnessCache:
    """Build the Redis freshness cache dependency for request handlers.

    The dependency is resolved lazily so non-cache endpoints remain available
    even when Redis-backed inspection is not being used.

    Args:
        None.

    Returns:
        PolicyFreshnessCache: Configured Redis freshness cache client.
    """

    return PolicyFreshnessCache(IndexerSettings.from_env())

from __future__ import annotations

from app.cache.policy_freshness_cache import PolicyFreshnessCache
from app.core.config import IndexerSettings
from app.ingestion.crawler import VisitorProgramCrawler
from app.vectorstore.index_manager import VisitorProgramIndexer


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


def get_indexer_manager() -> VisitorProgramIndexer:
    """Build the shared single-source indexer dependency.

    The dependency wires together environment-backed settings, the default
    crawler, Pinecone client, and Redis freshness cache so API handlers can
    execute direct indexing requests without duplicating service construction.

    Args:
        None.

    Returns:
        VisitorProgramIndexer: Fully configured indexer manager.
    """

    settings = IndexerSettings.from_env()
    return VisitorProgramIndexer(
        settings,
        crawler=VisitorProgramCrawler(),
        policy_cache=PolicyFreshnessCache(settings),
    )

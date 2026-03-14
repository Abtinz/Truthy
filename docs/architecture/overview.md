# Architecture Overview

Truthy is designed as a multi-service AI application platform for government application completeness verification.

## Primary Runtime Flow

1. The officer or analyst interacts with the `streamlit` frontend.
2. The frontend sends requests to the `api` service.
3. The `api` service orchestrates workflow execution and delegates specialized tasks.
4. The `indexer` service manages policy ingestion, parsing, normalization, and retrieval preparation.
5. The `agentic-rag` service handles retrieval, reasoning, issue explanation, and completeness analysis.
6. Shared infrastructure supports caching, retrieval, and service coordination.

## Design Principles

- The FastAPI service is the single main backend entrypoint.
- The downstream intelligence services remain specialized and independently scalable.
- Policy knowledge is treated as versioned system input.
- Explainability and auditability are part of the architecture, not optional add-ons.
- Human officers remain the final decision-makers.

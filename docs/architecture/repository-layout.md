# Repository Layout

This repository is intentionally shaped as a monorepo with strong service boundaries.

## Why This Layout

- It keeps the FastAPI orchestration layer separate from the reasoning and indexing engines.
- It allows the frontend to evolve independently from backend services.
- It creates a clean path for future scaling, testing, deployment, and ownership boundaries.

## Intended Ownership Model

- `services/api`: workflow orchestration and public backend interface
- `services/streamlit`: user experience and internal review workflows
- `services/indexer`: policy knowledge ingestion and indexing
- `services/agentic-rag`: retrieval, reasoning, and explainable findings
- `shared/*`: cross-service standards and reusable definitions
- `docs/*`: architecture and product communication
- `infra/*`: foundational infrastructure notes and deployment preparation

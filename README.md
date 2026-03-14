# Truthy

Truthy is a documentation-first monorepo for an AI-assisted government application completeness verification platform.

This repository is intentionally scaffolded without implementation code. The current goal is to lock the architecture, service boundaries, file layout, and local runtime topology before writing application logic.

## Services

- `api`: Main FastAPI gateway and orchestration layer. This is the system entrypoint and calls the downstream `indexer` and `agentic-rag` services.
- `streamlit`: Frontend for officers, analysts, and demo workflows.
- `indexer`: Policy ingestion and indexing service for program guides, checklists, form instructions, and rule updates.
- `agentic-rag`: Retrieval and reasoning service that analyzes extracted application data against indexed policy knowledge.

## Architecture Summary

The platform is organized as a service-oriented monorepo:

1. `streamlit` provides the user interface.
2. `api` receives UI requests and coordinates workflows.
3. `api` calls `indexer` for policy ingestion and refresh workflows.
4. `api` calls `agentic-rag` for retrieval, reasoning, and completeness analysis.
5. Shared contracts, prompts, and documentation live at the repository level.

This structure follows the requested operating model:

- frontend with Streamlit
- indexer service
- agentic RAG service
- FastAPI as the main service that invokes the two backend intelligence services

## Repository Docs

- [`STRUCTURE.md`](/Users/abtinzandi/Desktop/Truthy/STRUCTURE.md): Full folder-by-folder explanation.
- [`docs/architecture/overview.md`](/Users/abtinzandi/Desktop/Truthy/docs/architecture/overview.md): High-level system architecture.
- [`docs/architecture/service-interactions.md`](/Users/abtinzandi/Desktop/Truthy/docs/architecture/service-interactions.md): Service communication model.

## Docker Compose

The repository includes a `docker-compose.yml` file with the following services:

- `api`
- `streamlit`
- `indexer`
- `agentic-rag`
- `redis`
- `qdrant`

`redis` and `qdrant` are included as foundational infrastructure for caching and retrieval-oriented indexing.

## Indexer Novelty

The indexer now includes a Redis-backed IRCC freshness cache.

- For each operational-guidelines source URL, the crawler extracts only the page's `Date modified` value.
- Redis stores the source link identity and the last observed modified date.
- Before indexing, the indexer compares the current modified date against Redis.
- If the date has not changed, the source is skipped before chunking, embedding, and Pinecone upsert.

This reduces unnecessary embedding calls and makes policy-refresh runs cheaper when IRCC pages are unchanged.

## Local Run Instructions

1. Copy environment variables:

```bash
cp .env.example .env
```

2. Build the containers:

```bash
docker compose build
```

3. Start the stack:

```bash
docker compose up
```

4. Planned service endpoints:

- Streamlit UI: `http://localhost:8501`
- FastAPI gateway: `http://localhost:8000`
- Indexer service: `http://localhost:8001`
- Agentic RAG service: `http://localhost:8002`
- Qdrant: `http://localhost:6333`

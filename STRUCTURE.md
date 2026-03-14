# Repository Structure

This document explains the intended architecture and the purpose of each major repository area.

## Architectural Intent

The monorepo is organized around one primary backend gateway and two specialized AI backend services:

- `streamlit` is the user-facing frontend.
- `api` is the main FastAPI service and the system entrypoint.
- `indexer` is the policy ingestion and indexing backend.
- `agentic-rag` is the retrieval and reasoning backend.

The operational rule is simple:

- the frontend calls `api`
- `api` calls `indexer`
- `api` calls `agentic-rag`

## Repository Tree

```text
Truthy/
├── .env.example
├── docker-compose.yml
├── README.md
├── STRUCTURE.md
├── docs/
│   ├── architecture/
│   └── product/
├── infra/
│   ├── docker/
│   ├── qdrant/
│   └── redis/
├── shared/
│   ├── contracts/
│   ├── policies/
│   └── prompts/
└── services/
    ├── agentic-rag/
    ├── api/
    ├── indexer/
    └── streamlit/
```

## Top Level Files

- [`README.md`](/Users/abtinzandi/Desktop/Truthy/README.md): Main project overview, service summary, and Docker instructions.
- [`STRUCTURE.md`](/Users/abtinzandi/Desktop/Truthy/STRUCTURE.md): Repository map and architecture explanation.
- [`docker-compose.yml`](/Users/abtinzandi/Desktop/Truthy/docker-compose.yml): Local service orchestration definition.
- [`.env.example`](/Users/abtinzandi/Desktop/Truthy/.env.example): Shared environment variable template.

## Documentation Area

- [`docs/architecture/overview.md`](/Users/abtinzandi/Desktop/Truthy/docs/architecture/overview.md): Overall system architecture and design principles.
- [`docs/architecture/service-interactions.md`](/Users/abtinzandi/Desktop/Truthy/docs/architecture/service-interactions.md): Service boundaries and communication model.
- [`docs/product/vision.md`](/Users/abtinzandi/Desktop/Truthy/docs/product/vision.md): Product-level framing for future investor, partner, or internal strategy use.

## Shared Assets

- [`shared/contracts`](/Users/abtinzandi/Desktop/Truthy/shared/contracts): API and payload contract placeholders shared across services.
- [`shared/prompts`](/Users/abtinzandi/Desktop/Truthy/shared/prompts): Shared prompt assets for extraction, reasoning, and controlled generation.
- [`shared/policies`](/Users/abtinzandi/Desktop/Truthy/shared/policies): Common policy rule schemas and versioning placeholders.

## Infrastructure

- [`infra/docker/README.md`](/Users/abtinzandi/Desktop/Truthy/infra/docker/README.md): Shared container conventions.
- [`infra/redis/README.md`](/Users/abtinzandi/Desktop/Truthy/infra/redis/README.md): Redis usage placeholder for cache and transient state.
- [`infra/qdrant/README.md`](/Users/abtinzandi/Desktop/Truthy/infra/qdrant/README.md): Qdrant usage placeholder for retrieval infrastructure.

## Service Breakdown

### API Service

- [`services/api`](/Users/abtinzandi/Desktop/Truthy/services/api): Main FastAPI gateway.
- [`services/api/app/clients/indexer_client.py`](/Users/abtinzandi/Desktop/Truthy/services/api/app/clients/indexer_client.py): Planned outbound client for indexer communication.
- [`services/api/app/clients/agentic_rag_client.py`](/Users/abtinzandi/Desktop/Truthy/services/api/app/clients/agentic_rag_client.py): Planned outbound client for reasoning service communication.
- [`services/api/app/services/application_review.py`](/Users/abtinzandi/Desktop/Truthy/services/api/app/services/application_review.py): Planned workflow orchestration module.

### Streamlit Service

- [`services/streamlit`](/Users/abtinzandi/Desktop/Truthy/services/streamlit): Frontend workspace.
- [`services/streamlit/app.py`](/Users/abtinzandi/Desktop/Truthy/services/streamlit/app.py): Planned Streamlit entrypoint.
- [`services/streamlit/pages/upload.py`](/Users/abtinzandi/Desktop/Truthy/services/streamlit/pages/upload.py): Planned upload experience page.
- [`services/streamlit/pages/review_report.py`](/Users/abtinzandi/Desktop/Truthy/services/streamlit/pages/review_report.py): Planned analysis result page.

### Indexer Service

- [`services/indexer`](/Users/abtinzandi/Desktop/Truthy/services/indexer): Policy ingestion backend workspace.
- [`services/indexer/app/ingestion/crawler.py`](/Users/abtinzandi/Desktop/Truthy/services/indexer/app/ingestion/crawler.py): Planned source collection module.
- [`services/indexer/app/parsers/policy_parser.py`](/Users/abtinzandi/Desktop/Truthy/services/indexer/app/parsers/policy_parser.py): Planned normalization parser.
- [`services/indexer/app/rules/rule_builder.py`](/Users/abtinzandi/Desktop/Truthy/services/indexer/app/rules/rule_builder.py): Planned structured rule generation module.
- [`services/indexer/app/vectorstore/index_manager.py`](/Users/abtinzandi/Desktop/Truthy/services/indexer/app/vectorstore/index_manager.py): Planned retrieval indexing coordinator.

### Agentic RAG Service

- [`services/agentic-rag`](/Users/abtinzandi/Desktop/Truthy/services/agentic-rag): Retrieval and reasoning backend workspace.
- [`services/agentic-rag/app/agents/orchestrator.py`](/Users/abtinzandi/Desktop/Truthy/services/agentic-rag/app/agents/orchestrator.py): Planned multi-step reasoning controller.
- [`services/agentic-rag/app/retrieval/retriever.py`](/Users/abtinzandi/Desktop/Truthy/services/agentic-rag/app/retrieval/retriever.py): Planned evidence and rule retrieval module.
- [`services/agentic-rag/app/reasoning/completeness_checker.py`](/Users/abtinzandi/Desktop/Truthy/services/agentic-rag/app/reasoning/completeness_checker.py): Planned completeness validation logic boundary.
- [`services/agentic-rag/app/explanations/report_builder.py`](/Users/abtinzandi/Desktop/Truthy/services/agentic-rag/app/explanations/report_builder.py): Planned officer-facing explanation builder.

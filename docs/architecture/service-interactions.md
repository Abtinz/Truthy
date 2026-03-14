# Service Interactions

This document defines the intended interaction model across services.

## Main Control Plane

- `streamlit` calls `api`.
- `api` calls `indexer`.
- `api` calls `agentic-rag`.

## Indexer Responsibilities

- ingest official government program sources
- normalize policy content
- extract structured rules
- prepare searchable and retrievable knowledge assets
- refresh indexed knowledge when policies change

## Agentic RAG Responsibilities

- retrieve policy knowledge for a specific program
- reason over extracted application facts
- generate completeness and inconsistency findings
- produce explanation-ready evidence structures

## FastAPI Responsibilities

- expose stable API endpoints to the frontend
- coordinate workflows across backend services
- compose the final officer-facing response
- enforce service boundaries and safety constraints

## Frontend Responsibilities

- upload application files
- trigger analysis workflows
- display reports and evidence clearly
- keep the human reviewer in control

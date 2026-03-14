from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(title="Truthy Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return the basic runtime status for the agentic-RAG service.

    Args:
        None.

    Returns:
        dict[str, str]: Basic service availability payload.
    """

    return {"service": "truthy-agentic-rag", "status": "ok"}

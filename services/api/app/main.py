from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Truthy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "Truthy API",
        "status": "running",
    }


@app.get("/health")
def read_health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/review")
def create_review() -> dict[str, str]:
    return {
        "message": "Review endpoint connected",
    }


@app.get("/review/{review_id}")
def get_review(review_id: str) -> dict[str, str]:
    return {
        "review_id": review_id,
        "status": "pending",
    }


@app.post("/policy/refresh")
def refresh_policy() -> dict[str, str]:
    return {
        "message": "Policy refresh triggered",
    }

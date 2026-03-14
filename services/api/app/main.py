from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ProcessFileRequest(BaseModel):
    data: bytes
    name: str
    format: Literal["PDF"]


class ProcessRequest(BaseModel):
    application_name: str
    files: list[ProcessFileRequest]


class ProcessResponse(BaseModel):
    report: str


app = FastAPI(title="Truthy API")


@app.post("/process", response_model=ProcessResponse)
def process_application(payload: ProcessRequest) -> ProcessResponse:
    return ProcessResponse(report="Process endpoint connected")

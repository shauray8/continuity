import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any , AsyncGenerator , Optional

from fastapi import FastAPI , Request
from fastapi.responses import JSONResponse , Response , StreamingResponse


TIMEOUT_KEEP_ALIFVE = 5 
app = FastAPI()

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/generate")
async def generate(request: Request) -> Response:
    pass


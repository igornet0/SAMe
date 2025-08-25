from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse

from same_api.database import User
from .configuration import Server

import logging

router = APIRouter(tags=["Main"])

logger = logging.getLogger("app_fastapi.main")

@router.get("/")
async def read_root(request: Request):
    return {"message": "SAMe Analog Search API", "status": "active"} 


@router.get("/healthz")
async def healthz():
    """Lightweight health endpoint without external deps."""
    return {"status": "ok"}

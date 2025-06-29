from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Depends, status, Body, Request
from fastapi.security import (HTTPBearer,
                              OAuth2PasswordRequestForm)
from datetime import timedelta

from core import settings
from backend.app.configuration import (Server, )

import logging

http_bearer = HTTPBearer(auto_error=False)

router = APIRouter(prefix="/search", tags=["Search"])

logger = logging.getLogger("app_fastapi.search")

@router.get("/")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.get("/team_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "team.html",
        {"request": request}
    )

@router.get("/contact_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "contact.html",
        {"request": request}
    )

@router.get("/faq")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "faq.html",
        {"request": request}
    )

@router.get("/pricing_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "pricing.html",
        {"request": request}
    )
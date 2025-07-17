from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse

from core.database import User
from backend.app.configuration import Server

import logging

router = APIRouter(tags=["Main"])

logger = logging.getLogger("app_fastapi.main")

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

@router.get("/profile_page")
async def read_root():
    return RedirectResponse(url=Server.frontend_url)

@router.get("/pricing_page")
async def read_root(request: Request):
    return Server.templates.TemplateResponse(
        "pricing.html",
        {"request": request}
    )
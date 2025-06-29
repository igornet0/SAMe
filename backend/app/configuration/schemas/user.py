from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr
from datetime import datetime

class UserLoginResponse(BaseModel):
    login: Optional[str] = None
    email: Optional[str] = None
    password: str

class UserResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    login: str
    email: Optional[EmailStr] = None
    password: str
    balance: float
    created: datetime
    active: Optional[bool] = True

class TokenData(BaseModel):
    email: Optional[str] = None
    login: Optional[str] = None

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: Optional[str] = "Bearer"

    message: Optional[str] = None

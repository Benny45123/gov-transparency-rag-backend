import os
from typing import Optional
from enum import Enum

from fastapi import Depends,HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import Client,create_client
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or ""
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
JWT_SECRET=os.getenv("JWT_SECRET") or ""
JWT_ALGORITHM = "HS256" or ""

supabase: Client = create_client(SUPABASE_URL,SUPABASE_ANON_KEY)
supabase_admin: Client = create_client(SUPABASE_URL,SUPABASE_SERVICE_ROLE_KEY)

bearer_scheme = HTTPBearer(auto_error=False)

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class AuthenticatedUser:
    def __init__(self, user_id: str,email:str, role: UserRole):
        self.user_id = user_id
        self.email = email
        self.role = role
    def has_role(self, *roles: UserRole) -> bool:
        return self.role in roles

def verify_jwt(token: str) -> Optional[dict]:
    """Validate a Supabase access token and return auth claims used by the API."""
    try:
        result = supabase.auth.get_user(token)
        user = result.user
        if not user:
            return None
        return {
            "sub": user.id,
            "email": user.email or "",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
def get_user_role(user_id: str) -> UserRole:
    """Fetch role from user_roles table using service role client."""
    try:
        result = (
            supabase_admin.table("user_roles")
            .select("role")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if result.data and isinstance(result.data, dict):
            return UserRole(result.data.get("role", "user"))
        return UserRole.USER  # fallback
    except Exception:
        return UserRole.USER



async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme))->AuthenticatedUser:
    """
    Dependency — validates Bearer token and returns the current user with role.
    Usage: user: AuthenticatedUser = Depends(get_current_user)
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = verify_jwt(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    email = payload.get("email", "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = get_user_role(user_id)
    return AuthenticatedUser(user_id=user_id,email=email, role=role)

# Factory for role-based dependencies

def require_role(*roles: UserRole):
    """
    Factory that returns a dependency enforcing role membership.

    Usage:
        @app.post("/admin-only")
        async def admin_ep(user = Depends(require_role(UserRole.ADMIN))):
            ...
    """
    async def dependency(user: AuthenticatedUser = Depends(get_current_user)):
        if not user.has_role(*roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in roles]}",
            )
        return user
    return dependency

require_admin   = require_role(UserRole.ADMIN)
require_viewer  = require_role(UserRole.ADMIN, UserRole.USER)

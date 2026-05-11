import os
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from auth import supabase_admin,UserRole,get_user_role,AuthenticatedUser,get_current_user,require_role
from fastapi.responses import RedirectResponse
from supabase import create_client,Client
from pydantic import BaseModel

router = APIRouter(prefix="/auth",tags=["auth"])
FRONTEND_URL = os.getenv("FRONTEND_URL") or "http://localhost:3000"
SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or ""
supabase: Client = create_client(SUPABASE_URL ,SUPABASE_ANON_KEY )


def _url_join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"

@router.get("/login/google")
async def login_google():
    """
    Returns the Google OAuth URL. Frontend redirects user to this URL.
    Supabase handles the OAuth dance and redirects back to /auth/callback.
    """
    response= supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": _url_join(os.getenv("BACKEND_URL", "http://localhost:8000"), "/auth/callback")
        }
    })
    return {"url": response.url}

@router.get("/callback")
async def auth_callback(
    request: Request,
    code: str | None = None,
    error: str | None = None,
    error_code: str | None = None,
    error_description: str | None = None,
):
    """
    Supabase redirects here after Google login.
    Exchange code → session, then redirect to frontend with token.
    """
    if error or error_code or error_description:
        params = urlencode(
            {
                "error": error or "oauth_error",
                "error_code": error_code or "",
                "error_description": error_description or "Google OAuth failed before a session code was returned.",
            }
        )
        return RedirectResponse(url=f"{_url_join(FRONTEND_URL, '/auth/success')}?{params}")

    if not code:
        params = urlencode(
            {
                "error": "missing_code",
                "error_description": "OAuth callback did not include a session code.",
            }
        )
        return RedirectResponse(url=f"{_url_join(FRONTEND_URL, '/auth/success')}?{params}")

    try:
        session = supabase.auth.exchange_code_for_session({
            "auth_code": code,
            "code_verifier": "",
            "redirect_to": _url_join(os.getenv("BACKEND_URL", "http://localhost:8000"), "/auth/callback")
        })
        if not session or not session.session:
            raise HTTPException(status_code=400, detail="Failed to exchange code for session")
        access_token = session.session.access_token
        # Send token to frontend via URL fragment or query param
        return RedirectResponse(
            url=f"{_url_join(FRONTEND_URL, '/auth/success')}?token={access_token}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {e}")

@router.get("/me")
async def get_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    """
    Protected endpoint to get current user info.
    Frontend can call this to verify token and fetch user details.
    """
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "role": current_user.role,
    }
    
class RoleUpdateRequest(BaseModel):
    user_id: str
    role: UserRole


@router.put("/roles", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def update_role(body: RoleUpdateRequest):
    """Admin-only: update a user's role."""
    try:
        supabase_admin.table("user_roles").upsert({
            "user_id": body.user_id,
            "role": body.role.value,
        }).execute()
        return {"message": f"Role updated to {body.role.value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/roles", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def list_roles():
    """Admin-only: list all user roles."""
    result = supabase_admin.table("user_roles").select("*").execute()
    return result.data

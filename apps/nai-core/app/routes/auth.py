"""
nAI Core Auth Routes
Authentication and user management endpoints
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import Settings, get_config
from ..models.schemas import (
    UserCreate,
    UserLogin,
    User,
    Token,
)
from ..services.auth import AuthService, get_auth_service
from ..utils.logging import get_logger

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_config),
    auth: AuthService = Depends(get_auth_service),
) -> Optional[User]:
    """
    Dependency to get current user from JWT token.
    Returns None if auth is disabled or no token provided.
    """
    if not settings.auth_enabled:
        return None
    
    if credentials is None:
        return None
    
    user = auth.get_current_user(credentials.credentials)
    return user


async def require_auth(
    user: Optional[User] = Depends(get_current_user),
    settings: Settings = Depends(get_config),
) -> User:
    """
    Dependency that requires authentication.
    Raises 401 if not authenticated.
    """
    if not settings.auth_enabled:
        # Return a dummy user when auth is disabled
        return User(
            id="anonymous",
            username="anonymous",
            email="anonymous@local",
            is_active=True,
            is_admin=True,
            created_at="2024-01-01T00:00:00Z",
        )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def require_admin(
    user: User = Depends(require_auth),
    settings: Settings = Depends(get_config),
) -> User:
    """
    Dependency that requires admin privileges.
    """
    if not settings.auth_enabled:
        return user
    
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    
    return user


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    settings: Settings = Depends(get_config),
    auth: AuthService = Depends(get_auth_service),
) -> User:
    """
    Register a new user.
    
    Returns the created user (without password).
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=400,
            detail="Authentication is disabled. Set NAI_AUTH_ENABLED=true to enable."
        )
    
    user = auth.create_user(user_data)
    
    if user is None:
        raise HTTPException(
            status_code=400,
            detail="Username or email already exists",
        )
    
    return user


@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    settings: Settings = Depends(get_config),
    auth: AuthService = Depends(get_auth_service),
) -> Token:
    """
    Authenticate and get access token.
    
    Returns JWT token for API authentication.
    """
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=400,
            detail="Authentication is disabled. Set NAI_AUTH_ENABLED=true to enable."
        )
    
    user = auth.authenticate_user(credentials.username, credentials.password)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    token = auth.create_access_token(user)
    return token


@router.get("/me", response_model=User)
async def get_me(
    user: User = Depends(require_auth),
) -> User:
    """
    Get current authenticated user.
    """
    return user


@router.get("/users", response_model=list[User])
async def list_users(
    admin: User = Depends(require_admin),
    auth: AuthService = Depends(get_auth_service),
) -> list[User]:
    """
    List all users (admin only).
    """
    return auth.list_users()


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
    auth: AuthService = Depends(get_auth_service),
):
    """
    Delete a user (admin only).
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete yourself",
        )
    
    success = auth.delete_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    
    return {"success": True, "message": "User deleted"}


@router.post("/users/{user_id}/admin")
async def toggle_admin(
    user_id: str,
    is_admin: bool,
    admin: User = Depends(require_admin),
    auth: AuthService = Depends(get_auth_service),
):
    """
    Grant or revoke admin privileges (admin only).
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot modify your own admin status",
        )
    
    user = auth.update_user(user_id, is_admin=is_admin)
    
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    
    return {"success": True, "user": user}


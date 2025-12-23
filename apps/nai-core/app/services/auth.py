"""
nAI Core Authentication Service
JWT-based authentication with user management
"""

import os
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config import Settings, get_settings
from ..models.schemas import User, UserCreate, Token, TokenData
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """
    Service for authentication and user management.
    Uses file-based storage for simplicity (upgrade to DB in production).
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._users_file = os.path.join(settings.data_path, "users.json")
        self._ensure_storage()
    
    def _ensure_storage(self) -> None:
        """Ensure users file exists."""
        os.makedirs(os.path.dirname(self._users_file), exist_ok=True)
        if not os.path.exists(self._users_file):
            self._save_users({})
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from file."""
        try:
            with open(self._users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users(self, users: Dict[str, Dict[str, Any]]) -> None:
        """Save users to file."""
        with open(self._users_file, "w") as f:
            json.dump(users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return secrets.token_hex(8)
    
    def create_user(self, user_data: UserCreate) -> Optional[User]:
        """
        Create a new user.
        Returns None if username or email already exists.
        """
        users = self._load_users()
        
        # Check for existing username or email
        for user in users.values():
            if user["username"].lower() == user_data.username.lower():
                logger.warning(f"Username already exists: {user_data.username}")
                return None
            if user["email"].lower() == user_data.email.lower():
                logger.warning(f"Email already exists: {user_data.email}")
                return None
        
        # Create user
        user_id = self._generate_user_id()
        now = datetime.utcnow().isoformat() + "Z"
        
        user_record = {
            "id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": self._hash_password(user_data.password),
            "is_active": True,
            "is_admin": False,
            "created_at": now,
        }
        
        users[user_id] = user_record
        self._save_users(users)
        
        logger.info(f"Created user: {user_data.username}")
        
        return User(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            is_active=True,
            is_admin=False,
            created_at=now,
        )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user by username and password.
        Returns User if successful, None otherwise.
        """
        users = self._load_users()
        
        for user in users.values():
            if user["username"].lower() == username.lower():
                if not user.get("is_active", True):
                    logger.warning(f"Inactive user attempted login: {username}")
                    return None
                
                if self._verify_password(password, user["password_hash"]):
                    logger.info(f"User authenticated: {username}")
                    return User(
                        id=user["id"],
                        username=user["username"],
                        email=user["email"],
                        is_active=user.get("is_active", True),
                        is_admin=user.get("is_admin", False),
                        created_at=user["created_at"],
                    )
                else:
                    logger.warning(f"Invalid password for user: {username}")
                    return None
        
        logger.warning(f"User not found: {username}")
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        users = self._load_users()
        user = users.get(user_id)
        
        if user:
            return User(
                id=user["id"],
                username=user["username"],
                email=user["email"],
                is_active=user.get("is_active", True),
                is_admin=user.get("is_admin", False),
                created_at=user["created_at"],
            )
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        users = self._load_users()
        
        for user in users.values():
            if user["username"].lower() == username.lower():
                return User(
                    id=user["id"],
                    username=user["username"],
                    email=user["email"],
                    is_active=user.get("is_active", True),
                    is_admin=user.get("is_admin", False),
                    created_at=user["created_at"],
                )
        return None
    
    def create_access_token(self, user: User) -> Token:
        """
        Create JWT access token for user.
        """
        expire = datetime.utcnow() + timedelta(minutes=self.settings.auth_token_expire_minutes)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
        
        token = jwt.encode(
            payload,
            self.settings.auth_secret_key,
            algorithm=self.settings.auth_algorithm,
        )
        
        return Token(
            access_token=token,
            token_type="bearer",
            expires_in=self.settings.auth_token_expire_minutes * 60,
        )
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify JWT token and return token data.
        Returns None if token is invalid or expired.
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.auth_secret_key,
                algorithms=[self.settings.auth_algorithm],
            )
            
            user_id = payload.get("sub")
            username = payload.get("username")
            exp = payload.get("exp")
            
            if user_id is None or username is None:
                return None
            
            return TokenData(
                user_id=user_id,
                username=username,
                exp=exp,
            )
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def get_current_user(self, token: str) -> Optional[User]:
        """
        Get current user from token.
        """
        token_data = self.verify_token(token)
        if token_data is None:
            return None
        
        return self.get_user_by_id(token_data.user_id)
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID."""
        users = self._load_users()
        
        if user_id in users:
            del users[user_id]
            self._save_users(users)
            logger.info(f"Deleted user: {user_id}")
            return True
        
        return False
    
    def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None,
    ) -> Optional[User]:
        """Update user fields."""
        users = self._load_users()
        
        if user_id not in users:
            return None
        
        user = users[user_id]
        
        if email is not None:
            user["email"] = email
        if password is not None:
            user["password_hash"] = self._hash_password(password)
        if is_active is not None:
            user["is_active"] = is_active
        if is_admin is not None:
            user["is_admin"] = is_admin
        
        self._save_users(users)
        
        return User(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            is_active=user.get("is_active", True),
            is_admin=user.get("is_admin", False),
            created_at=user["created_at"],
        )
    
    def list_users(self) -> list[User]:
        """List all users."""
        users = self._load_users()
        return [
            User(
                id=u["id"],
                username=u["username"],
                email=u["email"],
                is_active=u.get("is_active", True),
                is_admin=u.get("is_admin", False),
                created_at=u["created_at"],
            )
            for u in users.values()
        ]


# Singleton instance
_auth_service_instance: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get or create the auth service singleton."""
    global _auth_service_instance
    if _auth_service_instance is None:
        _auth_service_instance = AuthService(get_settings())
    return _auth_service_instance


"""
Authentication service for JWT token management and user verification.
This module handles user authentication, token generation, and access control
for admin endpoints.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from ..config import settings

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer token scheme
security = HTTPBearer()

# Admin credentials (in production, store in database)
ADMIN_CREDENTIALS = {
    "admin": {
        "username": "admin",
        "password_hash": None,  # Will be set when needed
        "role": "admin",
        "permissions": ["read", "write", "delete", "manage"]
    }
}

def get_admin_password_hash():
    """Get admin password hash, creating it if needed."""
    if ADMIN_CREDENTIALS["admin"]["password_hash"] is None:
        ADMIN_CREDENTIALS["admin"]["password_hash"] = pwd_context.hash("admin123")
    return ADMIN_CREDENTIALS["admin"]["password_hash"]

class AuthService:
    """Authentication service for user management and JWT operations."""
    
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: User's username
            password: User's plain text password
        
        Returns:
            User data if authentication successful, None otherwise
        """
        try:
            # Check if user exists
            user = ADMIN_CREDENTIALS.get(username)
            if not user:
                logger.warning(f"Authentication failed: user {username} not found")
                return None
            
            # Verify password
            password_hash = user["password_hash"] or get_admin_password_hash()
            if not self.verify_password(password, password_hash):
                logger.warning(f"Authentication failed: invalid password for user {username}")
                return None
            
            # Return user data without password hash
            user_data = {
                "username": user["username"],
                "role": user["role"],
                "permissions": user["permissions"]
            }
            
            logger.info(f"User {username} authenticated successfully")
            return user_data
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token.
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time (optional)
        
        Returns:
            JWT token string
        """
        try:
            to_encode = data.copy()
            
            # Set expiration time
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({"exp": expire})
            
            # Create token
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            
            logger.debug(f"Created access token for user: {data.get('username')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token data if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("username")
            
            if username is None:
                return None
            
            # Check if user still exists
            user = ADMIN_CREDENTIALS.get(username)
            if not user:
                return None
            
            return {
                "username": username,
                "role": payload.get("role"),
                "permissions": payload.get("permissions", [])
            }
            
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def create_user_session(self, username: str, role: str, permissions: list) -> Dict[str, Any]:
        """Create user session data for token."""
        return {
            "username": username,
            "role": role,
            "permissions": permissions,
            "issued_at": datetime.now(timezone.utc).isoformat()
        }


# Global auth service instance
auth_service = AuthService()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
    
    Returns:
        Current user data
    
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Verify token
        user_data = auth_service.verify_token(token)
        if user_data is None:
            raise credentials_exception
        
        return user_data
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency to require admin role for protected endpoints.
    
    Args:
        current_user: Current authenticated user from get_current_user
    
    Returns:
        Current user data if admin
    
    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.get("role") != "admin":
        logger.warning(f"Access denied: User {current_user.get('username')} is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


def require_admin_role(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency that requires admin role.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        Current user data if admin
    
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


def require_permission(permission: str):
    """
    Factory function to create permission-checking dependencies.
    
    Args:
        permission: Required permission
    
    Returns:
        Dependency function that checks for the permission
    """
    def check_permission(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return Depends(check_permission)


# Optional user class for future database-backed authentication
class User:
    """User model for database-backed authentication (future enhancement)."""
    
    def __init__(self, username: str, email: str, role: str, permissions: list):
        self.username = username
        self.email = email
        self.role = role
        self.permissions = permissions
        self.created_at = datetime.now(timezone.utc)
        self.last_login = None
        self.is_active = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "permissions": self.permissions,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active
        }
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == "admin"


# Authentication utilities
def generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
    
    Returns:
        Tuple of (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    return True, "Password is strong"


def create_api_key(user_id: str, expires_days: int = 30) -> str:
    """
    Create API key for programmatic access (future enhancement).
    
    Args:
        user_id: User identifier
        expires_days: API key expiration in days
    
    Returns:
        API key string
    """
    data = {
        "user_id": user_id,
        "type": "api_key",
        "exp": datetime.now(timezone.utc) + timedelta(days=expires_days)
    }
    
    return auth_service.create_access_token(data, timedelta(days=expires_days))
"""
Authentication API routes for user login and token management.
This module provides endpoints for user authentication and JWT token operations.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import timedelta
import logging

from ..services.auth_service import auth_service, get_current_user
from ..core.cache_redis import get_redis_client, RedisClient

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict
    expires_in: int

class UserResponse(BaseModel):
    username: str
    role: str
    permissions: list

@router.post("/token", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    Authenticate user and return JWT access token.
    
    Args:
        login_data: Username and password
        redis_client: Redis client for session management
    
    Returns:
        JWT token and user information
    """
    try:
        # Authenticate user
        user_data = auth_service.authenticate_user(
            login_data.username, 
            login_data.password
        )
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create session data
        session_data = auth_service.create_user_session(
            user_data["username"],
            user_data["role"],
            user_data["permissions"]
        )
        
        # Generate JWT token
        access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
        access_token = auth_service.create_access_token(
            data=session_data,
            expires_delta=access_token_expires
        )
        
        # Store session in Redis for tracking
        session_key = f"session:{user_data['username']}"
        await redis_client.set_cache(
            session_key, 
            session_data, 
            auth_service.access_token_expire_minutes * 60
        )
        
        # Log successful login
        logger.info(f"User {user_data['username']} logged in successfully")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_data,
            expires_in=auth_service.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current user from JWT token
    
    Returns:
        Current user information
    """
    try:
        return UserResponse(
            username=current_user["username"],
            role=current_user["role"],
            permissions=current_user["permissions"]
        )
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.post("/logout")
async def logout(
    current_user: dict = Depends(get_current_user),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    Logout user and invalidate session.
    
    Args:
        current_user: Current authenticated user
        redis_client: Redis client for session management
    
    Returns:
        Logout confirmation
    """
    try:
        # Remove session from Redis
        session_key = f"session:{current_user['username']}"
        await redis_client.delete_cache(session_key)
        
        logger.info(f"User {current_user['username']} logged out successfully")
        
        return {
            "message": "Successfully logged out",
            "user": current_user["username"]
        }
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )

@router.post("/refresh")
async def refresh_token(
    current_user: dict = Depends(get_current_user),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """
    Refresh JWT token for authenticated user.
    
    Args:
        current_user: Current authenticated user
        redis_client: Redis client for session management
    
    Returns:
        New JWT token
    """
    try:
        # Create new session data
        session_data = auth_service.create_user_session(
            current_user["username"],
            current_user["role"],
            current_user["permissions"]
        )
        
        # Generate new JWT token
        access_token_expires = timedelta(minutes=auth_service.access_token_expire_minutes)
        access_token = auth_service.create_access_token(
            data=session_data,
            expires_delta=access_token_expires
        )
        
        # Update session in Redis
        session_key = f"session:{current_user['username']}"
        await redis_client.set_cache(
            session_key, 
            session_data, 
            auth_service.access_token_expire_minutes * 60
        )
        
        logger.info(f"Token refreshed for user {current_user['username']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": auth_service.access_token_expire_minutes * 60
        }
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        )

@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    Validate current JWT token.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        Token validation status
    """
    try:
        return {
            "valid": True,
            "user": {
                "username": current_user["username"],
                "role": current_user["role"],
                "permissions": current_user["permissions"]
            },
            "message": "Token is valid"
        }
        
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Health check for authentication service
@router.get("/health")
async def auth_health_check():
    """Health check for authentication service."""
    try:
        # Test auth service functionality
        test_hash = auth_service.get_password_hash("test")
        is_valid = auth_service.verify_password("test", test_hash)
        
        return {
            "status": "healthy",
            "service": "authentication",
            "password_hashing": "functional" if is_valid else "error",
            "jwt_algorithm": auth_service.algorithm
        }
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e)
        }
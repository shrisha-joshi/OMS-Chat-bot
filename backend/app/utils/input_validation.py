"""
Input validation utilities for API endpoints.
Provides comprehensive validation for user inputs to prevent injection attacks,
DoS, and malformed data processing.
"""

import re
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

# Constants
MAX_QUERY_LENGTH = 10000  # 10K characters
MAX_FILENAME_LENGTH = 255
MAX_FILE_SIZE_BYTES = 209715200  # 200MB
ALLOWED_FILE_EXTENSIONS = {
    '.pdf', '.docx', '.txt', '.html', '.json', '.csv', '.xlsx',
    '.doc', '.rtf', '.md', '.xml'
}

# Dangerous patterns
DANGEROUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # XSS attempts
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',  # Event handlers
    r'\.\./',  # Path traversal
    r'\.\.\\',  # Path traversal (Windows)
]

class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Validate chat query input.
    
    Args:
        query: User query string
        max_length: Maximum allowed length
    
    Returns:
        Sanitized query string
    
    Raises:
        HTTPException: If validation fails
    """
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if not isinstance(query, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be a string"
        )
    
    # Remove leading/trailing whitespace
    query = query.strip()
    
    # Check length
    if len(query) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query too long. Maximum {max_length} characters allowed, got {len(query)}"
        )
    
    if len(query) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty after trimming whitespace"
        )
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Blocked potentially dangerous query pattern: {pattern}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query contains potentially dangerous content"
            )
    
    return query


def validate_filename(filename: str) -> str:
    """
    Validate and sanitize filename for upload.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    
    Raises:
        HTTPException: If validation fails
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty"
        )
    
    # Use only basename (prevents path traversal)
    import os
    filename = os.path.basename(filename)
    
    # Check length
    if len(filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filename too long. Maximum {MAX_FILENAME_LENGTH} characters allowed"
        )
    
    # Check for dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0', '\n', '\r']
    if any(char in filename for char in dangerous_chars):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename contains invalid characters"
        )
    
    # Check file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' not allowed. Allowed types: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
        )
    
    return filename


def validate_file_size(content_length: Optional[int], max_size: int = MAX_FILE_SIZE_BYTES) -> None:
    """
    Validate file size before processing.
    
    Args:
        content_length: Content-Length header value
        max_size: Maximum allowed size in bytes
    
    Raises:
        HTTPException: If validation fails
    """
    if content_length is None:
        raise HTTPException(
            status_code=status.HTTP_411_LENGTH_REQUIRED,
            detail="Content-Length header required for file uploads"
        )
    
    if content_length > max_size:
        max_size_mb = max_size / 1024 / 1024
        actual_size_mb = content_length / 1024 / 1024
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum {max_size_mb:.1f}MB allowed, got {actual_size_mb:.1f}MB"
        )
    
    if content_length == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File cannot be empty"
        )


def validate_document_id(doc_id: str) -> str:
    """
    Validate MongoDB document ID format.
    
    Args:
        doc_id: Document ID string
    
    Returns:
        Validated document ID
    
    Raises:
        HTTPException: If validation fails
    """
    if not doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document ID cannot be empty"
        )
    
    # MongoDB ObjectId is 24 hex characters
    if not re.match(r'^[a-f0-9]{24}$', doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    
    return doc_id


def validate_pagination(skip: int = 0, limit: int = 10, max_limit: int = 100) -> tuple[int, int]:
    """
    Validate pagination parameters.
    
    Args:
        skip: Number of items to skip
        limit: Number of items to return
        max_limit: Maximum allowed limit
    
    Returns:
        Tuple of validated (skip, limit)
    
    Raises:
        HTTPException: If validation fails
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter cannot be negative"
        )
    
    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit parameter must be at least 1"
        )
    
    if limit > max_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Limit parameter cannot exceed {max_limit}"
        )
    
    return skip, limit


def sanitize_dict(data: Dict[str, Any], allowed_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Sanitize dictionary by removing unknown keys.
    
    Args:
        data: Input dictionary
        allowed_keys: List of allowed keys (None = allow all)
    
    Returns:
        Sanitized dictionary
    """
    if allowed_keys is None:
        return data
    
    return {k: v for k, v in data.items() if k in allowed_keys}


def validate_email(email: str) -> str:
    """
    Validate email address format.
    
    Args:
        email: Email address
    
    Returns:
        Validated email
    
    Raises:
        HTTPException: If validation fails
    """
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email cannot be empty"
        )
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    return email.lower()


def validate_password_strength(password: str, min_length: int = 16) -> None:
    """
    Validate password strength.
    
    Args:
        password: Password string
        min_length: Minimum required length
    
    Raises:
        HTTPException: If password is too weak
    """
    if len(password) < min_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password must be at least {min_length} characters long"
        )
    
    # Check complexity
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
    
    if not (has_upper and has_lower and has_digit and has_special):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain uppercase, lowercase, digit, and special character"
        )


# Export all validation functions
__all__ = [
    'validate_query',
    'validate_filename',
    'validate_file_size',
    'validate_document_id',
    'validate_pagination',
    'sanitize_dict',
    'validate_email',
    'validate_password_strength',
    'ValidationError'
]

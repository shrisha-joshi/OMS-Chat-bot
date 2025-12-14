"""
PII Masking Utility for Production Logging
Prevents sensitive data from being logged

Usage:
    from app.utils.pii_masking import mask_pii
    
    safe_query = mask_pii(user_query)
    logger.info(f"Query: {safe_query}")
"""

import re
from typing import Dict, Any


def mask_pii(text: str) -> str:
    """
    Mask personally identifiable information (PII) in text.
    
    Patterns masked:
    - Email addresses: user@example.com → ***@***.***
    - Phone numbers: (555) 123-4567 → (***) ***-****
    - SSN: 123-45-6789 → ***-**-****
    - Credit cards: 1234-5678-9012-3456 → ****-****-****-****
    - IP addresses: 192.168.1.1 → ***.***.***.***
    
    Args:
        text: Input text to mask
    
    Returns:
        Text with PII masked
    """
    if not text:
        return text
    
    # Email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '***@***.***',
        text
    )
    
    # Phone numbers (various formats)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '***-***-****', text)
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '(***) ***-****', text)
    
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****', text)
    
    # Credit card numbers
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '****-****-****-****', text)
    
    # IP addresses (v4)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '***.***.***.***', text)
    
    return text


def mask_password_fields(data: dict) -> dict:
    """
    Recursively mask password fields in dictionaries.
    
    Args:
        data: Dictionary that may contain password fields
    
    Returns:
        Dictionary with passwords masked
    """
    if not isinstance(data, dict):
        return data
    
    masked = data.copy()
    password_keys = ['password', 'passwd', 'pwd', 'secret', 'token', 'api_key', 'apikey']
    
    for key, value in masked.items():
        if key.lower() in password_keys:
            masked[key] = '***MASKED***'
        elif isinstance(value, dict):
            masked[key] = mask_password_fields(value)
        elif isinstance(value, list):
            masked[key] = [mask_password_fields(item) if isinstance(item, dict) else item for item in value]
    
    return masked

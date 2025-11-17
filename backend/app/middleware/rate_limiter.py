"""
Rate limiting middleware for FastAPI endpoints.
Prevents abuse and protects backend resources.
"""

import time
import logging
from typing import Dict, Tuple
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta

from ..config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter based on IP address."""
    
    def __init__(self, app):
        super().__init__(app)
        # Store: {ip: [(timestamp, count)]}
        self.requests: Dict[str, list] = defaultdict(list)
        self.enabled = settings.rate_limit_enabled
        self.max_requests = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window_seconds
        
        # Exempt paths (health checks, websockets, etc.)
        self.exempt_paths = {"/health", "/docs", "/openapi.json", "/ws"}
        
        logger.info(f"Rate limiter initialized: {self.max_requests} req/{self.window_seconds}s")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip if disabled or exempt path
        if not self.enabled or any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old entries and check rate
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Remove expired entries
        self.requests[client_ip] = [
            timestamp for timestamp in self.requests[client_ip]
            if timestamp > cutoff_time
        ]
        
        # Check if rate limit exceeded
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}: {len(self.requests[client_ip])} requests")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds."
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.max_requests - len(self.requests[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(cutoff_time + self.window_seconds))
        
        return response
    
    def cleanup(self):
        """Clean up old entries (call periodically)."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                timestamp for timestamp in self.requests[ip]
                if timestamp > cutoff_time
            ]
            if not self.requests[ip]:
                del self.requests[ip]

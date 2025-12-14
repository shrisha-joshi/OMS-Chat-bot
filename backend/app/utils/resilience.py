"""
Resilience utilities for API and service reliability.

Implements:
- Circuit Breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Rate limiting
- Timeout management
- Health check decorators

Research Reference: Microservices resilience patterns from production systems
"""

import asyncio
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Track circuit breaker metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def record_success(self):
        """Record successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)
    
    def record_failure(self):
        """Record failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(timezone.utc)
    
    def record_rejection(self):
        """Record rejected call."""
        self.rejected_calls += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": self.get_success_rate(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None
        }


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by temporarily blocking calls to failing services.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing) → CLOSED
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Consecutive failures before opening
            success_threshold: Consecutive successes in half-open to close
            timeout: Seconds before trying half-open from open
            half_open_timeout: Seconds before going back to open from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_timeout = half_open_timeout
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.opened_at: Optional[datetime] = None
        self.half_opened_at: Optional[datetime] = None
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, timeout={timeout}s"
        )
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset from OPEN to HALF_OPEN."""
        if self.state != CircuitState.OPEN or not self.opened_at:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.opened_at).total_seconds()
        return elapsed >= self.timeout
    
    def _should_timeout_half_open(self) -> bool:
        """Check if half-open state has timed out."""
        if self.state != CircuitState.HALF_OPEN or not self.half_opened_at:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.half_opened_at).total_seconds()
        return elapsed >= self.half_open_timeout
    
    async def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            RuntimeError: If circuit is open
            Exception: Original exception from function
        """
        # Check state transitions
        if self._should_timeout_half_open():
            logger.warning(f"Circuit '{self.name}' half-open timeout, reopening")
            self._open()
        elif self._should_attempt_reset():
            logger.info(f"Circuit '{self.name}' attempting reset to half-open")
            self._half_open()
        
        # Check if should reject call
        if self.state == CircuitState.OPEN:
            self.metrics.record_rejection()
            raise RuntimeError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Failed {self.metrics.consecutive_failures} times. "
                f"Will retry in {self.timeout}s"
            )
        
        # Execute call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.metrics.record_success()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.metrics.consecutive_successes >= self.success_threshold:
                logger.info(
                    f"✅ Circuit '{self.name}' recovered after "
                    f"{self.metrics.consecutive_successes} successes"
                )
                self._close()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.metrics.consecutive_failures = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        self.metrics.record_failure()
        
        logger.warning(
            f"Circuit '{self.name}' failure #{self.metrics.consecutive_failures}: {exception}"
        )
        
        if self.state == CircuitState.HALF_OPEN:
            # Go back to open on any failure in half-open
            logger.warning(f"Circuit '{self.name}' failed in half-open, reopening")
            self._open()
        elif self.state == CircuitState.CLOSED:
            if self.metrics.consecutive_failures >= self.failure_threshold:
                logger.error(
                    f"⚠️ Circuit '{self.name}' opened after "
                    f"{self.metrics.consecutive_failures} consecutive failures"
                )
                self._open()
    
    def _close(self):
        """Close circuit (normal operation)."""
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self.half_opened_at = None
        self.metrics.consecutive_failures = 0
    
    def _open(self):
        """Open circuit (reject calls)."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now(timezone.utc)
        self.half_opened_at = None
    
    def _half_open(self):
        """Half-open circuit (test recovery)."""
        self.state = CircuitState.HALF_OPEN
        self.half_opened_at = datetime.now(timezone.utc)
        self.metrics.consecutive_successes = 0
    
    def get_state(self) -> str:
        """Get current state."""
        return self.state.value
    
    def get_metrics(self) -> dict:
        """Get metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            **self.metrics.to_dict()
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        logger.info(f"Manually resetting circuit '{self.name}'")
        self._close()
        self.metrics = CircuitBreakerMetrics()


# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0
) -> CircuitBreaker:
    """
    Get or create circuit breaker.
    
    Args:
        name: Circuit breaker identifier
        failure_threshold: Consecutive failures before opening
        timeout: Seconds before retry
    
    Returns:
        Circuit breaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            timeout=timeout
        )
    return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    timeout: float = 60.0
):
    """
    Decorator for circuit breaker protection.
    
    Args:
        name: Circuit breaker identifier (defaults to function name)
        failure_threshold: Consecutive failures before opening
        timeout: Seconds before retry
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        breaker_name = name or func.__name__
        breaker = get_circuit_breaker(breaker_name, failure_threshold, timeout)
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return asyncio.run(breaker.call(func, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def _calculate_backoff_delay(attempt: int, initial_delay: float, exponential_base: float, max_delay: float) -> float:
    """Calculate exponential backoff delay."""
    return min(initial_delay * (exponential_base ** attempt), max_delay)


def _log_retry_attempt(func_name: str, attempt: int, max_attempts: int, error: Exception, delay: float):
    """Log retry attempt information."""
    logger.warning(
        f"Function {func_name} attempt {attempt + 1}/{max_attempts} "
        f"failed: {error}. Retrying in {delay:.1f}s..."
    )


def _log_final_failure(func_name: str, max_attempts: int, error: Exception):
    """Log final failure after all retries exhausted."""
    logger.error(f"Function {func_name} failed after {max_attempts} attempts: {error}")


def _create_async_retry_wrapper(func, max_attempts, initial_delay, exponential_base, max_delay, exceptions):
    """Create async wrapper with retry logic."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_attempts - 1:
                    _log_final_failure(func.__name__, max_attempts, e)
                    raise
                
                delay = _calculate_backoff_delay(attempt, initial_delay, exponential_base, max_delay)
                _log_retry_attempt(func.__name__, attempt, max_attempts, e, delay)
                await asyncio.sleep(delay)
    return async_wrapper


def _create_sync_retry_wrapper(func, max_attempts, initial_delay, exponential_base, max_delay, exceptions):
    """Create sync wrapper with retry logic."""
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_attempts - 1:
                    _log_final_failure(func.__name__, max_attempts, e)
                    raise
                
                delay = _calculate_backoff_delay(attempt, initial_delay, exponential_base, max_delay)
                _log_retry_attempt(func.__name__, attempt, max_attempts, e, delay)
                time.sleep(delay)
    return sync_wrapper


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(func):
            return _create_async_retry_wrapper(func, max_attempts, initial_delay, exponential_base, max_delay, exceptions)
        else:
            return _create_sync_retry_wrapper(func, max_attempts, initial_delay, exponential_base, max_delay, exceptions)
    
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Limits the rate of operations using a token bucket algorithm.
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            capacity: Maximum token bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if acquired, False if rate limit exceeded
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, tokens: int = 1):
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while not await self.acquire(tokens):
            wait_time = tokens / self.rate
            await asyncio.sleep(wait_time)


def rate_limit(rate: float, capacity: int):
    """
    Decorator for rate limiting.
    
    Args:
        rate: Operations per second
        capacity: Burst capacity
    """
    limiter = RateLimiter(rate, capacity)
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            await limiter.wait_for_token()
            return await func(*args, **kwargs)
        
        return async_wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorator for timeout protection.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Operation timed out after {seconds}s")
        
        return async_wrapper
    
    return decorator


def get_all_circuit_breaker_metrics() -> list[dict]:
    """Get metrics for all circuit breakers."""
    return [breaker.get_metrics() for breaker in _circuit_breakers.values()]


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    for breaker in _circuit_breakers.values():
        breaker.reset()

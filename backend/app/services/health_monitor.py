"""
Connection Health Monitor Service.

Monitors health of all database connections and services:
- Periodic health checks
- Automatic reconnection attempts
- Health status reporting
- Alerting on failures

Research Reference: Production monitoring patterns for distributed systems
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Callable, Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ServiceHealth:
    """Track service health over time."""
    service_name: str
    current_status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_healthy: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    average_response_time_ms: float = 0.0
    
    def update(self, result: HealthCheckResult):
        """Update health with check result."""
        self.current_status = result.status
        self.last_check = result.timestamp
        self.total_checks += 1
        
        # Update response time (moving average)
        if self.average_response_time_ms == 0:
            self.average_response_time_ms = result.response_time_ms
        else:
            # Exponential moving average
            alpha = 0.3
            self.average_response_time_ms = (
                alpha * result.response_time_ms + 
                (1 - alpha) * self.average_response_time_ms
            )
        
        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.last_healthy = result.timestamp
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.total_failures += 1
    
    def get_uptime_percentage(self) -> float:
        """Calculate uptime percentage."""
        if self.total_checks == 0:
            return 0.0
        successful_checks = self.total_checks - self.total_failures
        return (successful_checks / self.total_checks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "current_status": self.current_status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_checks": self.total_checks,
            "total_failures": self.total_failures,
            "uptime_percentage": self.get_uptime_percentage(),
            "average_response_time_ms": round(self.average_response_time_ms, 2)
        }


class ConnectionHealthMonitor:
    """
    Monitor health of all database connections and services.
    
    Features:
    - Periodic health checks
    - Automatic reconnection
    - Health status tracking
    - Alerting
    """
    
    def __init__(
        self,
        check_interval: int = 30,
        unhealthy_threshold: int = 3,
        reconnect_attempts: int = 3
    ):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            unhealthy_threshold: Consecutive failures before unhealthy
            reconnect_attempts: Maximum reconnection attempts
        """
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.reconnect_attempts = reconnect_attempts
        
        # Service health tracking
        self.services: Dict[str, ServiceHealth] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.reconnect_handlers: Dict[str, Callable] = {}
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Callbacks
        self._on_unhealthy_callbacks: List[Callable] = []
        self._on_recovered_callbacks: List[Callable] = []
        
        logger.info(
            f"Health monitor initialized: check_interval={check_interval}s, "
            f"threshold={unhealthy_threshold}"
        )
    
    def register_service(
        self,
        service_name: str,
        health_check: Callable,
        reconnect_handler: Optional[Callable] = None
    ):
        """
        Register service for monitoring.
        
        Args:
            service_name: Service identifier
            health_check: Async function that returns bool (healthy)
            reconnect_handler: Optional async function to reconnect
        """
        self.services[service_name] = ServiceHealth(service_name=service_name)
        self.health_checks[service_name] = health_check
        
        if reconnect_handler:
            self.reconnect_handlers[service_name] = reconnect_handler
        
        logger.info(f"Registered service for monitoring: {service_name}")
    
    def on_service_unhealthy(self, callback: Callable):
        """Register callback for when service becomes unhealthy."""
        self._on_unhealthy_callbacks.append(callback)
    
    def on_service_recovered(self, callback: Callable):
        """Register callback for when service recovers."""
        self._on_recovered_callbacks.append(callback)
    
    async def check_service_health(self, service_name: str) -> HealthCheckResult:
        """
        Perform health check for a service.
        
        Args:
            service_name: Service identifier
        
        Returns:
            Health check result
        """
        if service_name not in self.health_checks:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                error_message="Service not registered"
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            health_check = self.health_checks[service_name]
            
            # Execute health check with timeout
            is_healthy = await asyncio.wait_for(
                health_check(),
                timeout=10.0
            )
            
            response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            
            return HealthCheckResult(
                service_name=service_name,
                status=status,
                response_time_ms=response_time_ms
            )
            
        except asyncio.TimeoutError:
            response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                error_message="Health check timeout"
            )
        
        except Exception as e:
            response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(f"Health check failed for {service_name}: {e}")
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def check_all_services(self) -> List[HealthCheckResult]:
        """
        Check health of all registered services.
        
        Returns:
            List of health check results
        """
        tasks = [
            self.check_service_health(service_name)
            for service_name in self.services.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check exception: {result}")
                continue
            health_results.append(result)
        
        # Update service health tracking
        for result in health_results:
            await self._update_service_health(result)
        
        return health_results
    
    async def _update_service_health(self, result: HealthCheckResult):
        """Update service health and trigger callbacks."""
        if result.service_name not in self.services:
            return
        
        service = self.services[result.service_name]
        previous_status = service.current_status
        
        # Update health
        service.update(result)
        
        # Check for status changes
        if previous_status == HealthStatus.HEALTHY and result.status == HealthStatus.UNHEALTHY:
            logger.warning(
                f"⚠️ Service {result.service_name} became unhealthy: {result.error_message}"
            )
            await self._trigger_unhealthy_callbacks(result)
            
        elif previous_status == HealthStatus.UNHEALTHY and result.status == HealthStatus.HEALTHY:
            logger.info(f"✅ Service {result.service_name} recovered")
            await self._trigger_recovered_callbacks(result)
        
        # Attempt reconnection if unhealthy threshold reached
        if service.consecutive_failures >= self.unhealthy_threshold:
            if result.service_name in self.reconnect_handlers:
                await self._attempt_reconnection(result.service_name)
    
    async def _attempt_reconnection(self, service_name: str):
        """Attempt to reconnect to unhealthy service."""
        if service_name not in self.reconnect_handlers:
            return
        
        logger.info(f"Attempting reconnection to {service_name}...")
        
        reconnect_handler = self.reconnect_handlers[service_name]
        
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                success = await asyncio.wait_for(
                    reconnect_handler(),
                    timeout=30.0
                )
                
                if success:
                    logger.info(f"✅ Successfully reconnected to {service_name}")
                    return
                
            except Exception as e:
                logger.error(
                    f"Reconnection attempt {attempt}/{self.reconnect_attempts} "
                    f"failed for {service_name}: {e}"
                )
            
            if attempt < self.reconnect_attempts:
                await asyncio.sleep(5 * attempt)  # Exponential backoff
        
        logger.error(
            f"❌ Failed to reconnect to {service_name} after "
            f"{self.reconnect_attempts} attempts"
        )
    
    async def _trigger_unhealthy_callbacks(self, result: HealthCheckResult):
        """Trigger callbacks for unhealthy service."""
        for callback in self._on_unhealthy_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Unhealthy callback error: {e}")
    
    async def _trigger_recovered_callbacks(self, result: HealthCheckResult):
        """Trigger callbacks for recovered service."""
        for callback in self._on_recovered_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Recovered callback error: {e}")
    
    async def start(self):
        """Start health monitoring."""
        if self._monitor_task is not None:
            logger.warning("Health monitor already running")
            return
        
        logger.info("Starting connection health monitor...")
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop health monitoring."""
        logger.info("Stopping connection health monitor...")
        self._shutdown_event.set()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Perform health checks
                    results = await self.check_all_services()
                    
                    # Log summary
                    healthy_count = sum(
                        1 for r in results if r.status == HealthStatus.HEALTHY
                    )
                    
                    if healthy_count == len(results):
                        logger.debug(
                            f"All {len(results)} services healthy "
                            f"(avg response: {sum(r.response_time_ms for r in results) / len(results):.1f}ms)"
                        )
                    else:
                        unhealthy = [
                            r.service_name for r in results 
                            if r.status != HealthStatus.HEALTHY
                        ]
                        logger.warning(
                            f"Health check: {healthy_count}/{len(results)} healthy. "
                            f"Unhealthy: {', '.join(unhealthy)}"
                        )
                    
                except Exception as e:
                    logger.error(f"Monitor loop error: {e}")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.debug("Monitor loop cancelled")
    
    def get_service_health(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get health status for a service."""
        if service_name not in self.services:
            return None
        return self.services[service_name].to_dict()
    
    def get_all_service_health(self) -> List[Dict[str, Any]]:
        """Get health status for all services."""
        return [service.to_dict() for service in self.services.values()]
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        if not self.services:
            return {
                "status": "unknown",
                "services": [],
                "healthy_count": 0,
                "total_count": 0
            }
        
        service_statuses = list(self.services.values())
        healthy_count = sum(
            1 for s in service_statuses 
            if s.current_status == HealthStatus.HEALTHY
        )
        
        # Determine overall status
        if healthy_count == len(service_statuses):
            overall_status = "healthy"
        elif healthy_count == 0:
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "healthy_count": healthy_count,
            "total_count": len(service_statuses),
            "services": [s.to_dict() for s in service_statuses]
        }


# Global health monitor instance
health_monitor = ConnectionHealthMonitor()


async def get_health_monitor() -> ConnectionHealthMonitor:
    """Dependency injection for health monitor."""
    return health_monitor

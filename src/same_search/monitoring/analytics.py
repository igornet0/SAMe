"""
Enterprise monitoring and analytics system for SAMe
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import uuid
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None


class MetricCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            metric = Metric(name, self.counters[key], MetricType.COUNTER, time.time(), labels)
            self.metrics[key].append(metric)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            metric = Metric(name, value, MetricType.GAUGE, time.time(), labels)
            self.metrics[key].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values (last 1000)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = Metric(name, value, MetricType.HISTOGRAM, time.time(), labels)
            self.metrics[key].append(metric)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record timer metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
            
            # Keep only recent values (last 1000)
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            
            metric = Metric(name, duration, MetricType.TIMER, time.time(), labels)
            self.metrics[key].append(metric)
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get current counter value"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current gauge value"""
        key = self._make_key(name, labels)
        return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
    
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics"""
        key = self._make_key(name, labels)
        values = self.timers.get(key, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'total_time': sum(values),
            'min_time': min(values),
            'max_time': max(values),
            'avg_time': statistics.mean(values),
            'median_time': statistics.median(values),
            'p95_time': self._percentile(values, 0.95),
            'p99_time': self._percentile(values, 0.99)
        }
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key from name and labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            for key, metric_deque in self.metrics.items():
                # Remove old metrics
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_alert_rule(
        self, 
        metric_name: str, 
        condition: str, 
        threshold: float, 
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = None
    ):
        """Add alert rule"""
        rule = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'level': level,
            'message_template': message_template or f"{metric_name} {condition} {threshold}"
        }
        self.alert_rules.append(rule)
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler function"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules"""
        with self._lock:
            for rule in self.alert_rules:
                metric_name = rule['metric_name']
                condition = rule['condition']
                threshold = rule['threshold']
                level = rule['level']
                
                # Get current metric value
                current_value = self._get_metric_value(metrics, metric_name)
                if current_value is None:
                    continue
                
                # Check condition
                alert_triggered = False
                if condition == 'gt' and current_value > threshold:
                    alert_triggered = True
                elif condition == 'lt' and current_value < threshold:
                    alert_triggered = True
                elif condition == 'eq' and abs(current_value - threshold) < 0.001:
                    alert_triggered = True
                
                alert_id = f"{metric_name}_{condition}_{threshold}"
                
                if alert_triggered:
                    if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                        # Create new alert
                        alert = Alert(
                            alert_id=alert_id,
                            level=level,
                            message=rule['message_template'].format(
                                metric_name=metric_name,
                                current_value=current_value,
                                threshold=threshold
                            ),
                            timestamp=time.time(),
                            metric_name=metric_name,
                            threshold_value=threshold,
                            current_value=current_value
                        )
                        
                        self.alerts[alert_id] = alert
                        self._notify_handlers(alert)
                
                else:
                    # Resolve alert if it exists
                    if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                        self.alerts[alert_id].resolved = True
                        self.alerts[alert_id].resolved_at = time.time()
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from metrics dict"""
        # Handle nested metric names like "api.response_time.avg"
        parts = metric_name.split('.')
        value = metrics
        
        try:
            for part in parts:
                value = value[part]
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None
    
    def _notify_handlers(self, alert: Alert):
        """Notify all registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts"""
        return list(self.alerts.values())


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default alert handler
        self.alert_manager.register_alert_handler(self._default_alert_handler)
    
    def _setup_default_alerts(self):
        """Setup default system alert rules"""
        # High response time alert
        self.alert_manager.add_alert_rule(
            'api.response_time.avg',
            'gt',
            1.0,  # 1 second
            AlertLevel.WARNING,
            "High API response time: {current_value:.2f}s (threshold: {threshold}s)"
        )
        
        # High error rate alert
        self.alert_manager.add_alert_rule(
            'api.error_rate',
            'gt',
            0.05,  # 5%
            AlertLevel.ERROR,
            "High API error rate: {current_value:.1%} (threshold: {threshold:.1%})"
        )
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(
            'system.memory_usage_percent',
            'gt',
            85.0,  # 85%
            AlertLevel.WARNING,
            "High memory usage: {current_value:.1f}% (threshold: {threshold}%)"
        )
    
    async def _default_alert_handler(self, alert: Alert):
        """Default alert handler - logs alerts"""
        level_map = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        
        logger.log(level_map[alert.level], f"ALERT [{alert.level.value.upper()}]: {alert.message}")
    
    async def start_monitoring(self, interval: float = 10.0):
        """Start continuous monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        if not self._running:
            return
        
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self._running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Get all current metrics
                current_metrics = self.get_all_metrics()
                
                # Check alerts
                self.alert_manager.check_alerts(current_metrics)
                
                # Cleanup old metrics
                self.metric_collector.cleanup_old_metrics()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            import os
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metric_collector.record_gauge('system.cpu_usage_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metric_collector.record_gauge('system.memory_usage_percent', memory.percent)
            self.metric_collector.record_gauge('system.memory_used_mb', memory.used / 1024 / 1024)
            self.metric_collector.record_gauge('system.memory_available_mb', memory.available / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metric_collector.record_gauge('system.disk_usage_percent', disk_percent)
            
            # Process info
            process = psutil.Process(os.getpid())
            self.metric_collector.record_gauge('process.memory_mb', process.memory_info().rss / 1024 / 1024)
            self.metric_collector.record_gauge('process.cpu_percent', process.cpu_percent())
            
        except ImportError:
            logger.debug("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        metrics = {
            'counters': {},
            'gauges': {},
            'histograms': {},
            'timers': {},
            'system': {
                'timestamp': time.time(),
                'uptime': time.time() - getattr(self, '_start_time', time.time())
            }
        }
        
        # Get counter values
        for key, value in self.metric_collector.counters.items():
            metrics['counters'][key] = value
        
        # Get gauge values
        for key, value in self.metric_collector.gauges.items():
            metrics['gauges'][key] = value
        
        # Get histogram stats
        for key in self.metric_collector.histograms.keys():
            name = key.split('{')[0]  # Remove labels for grouping
            metrics['histograms'][name] = self.metric_collector.get_histogram_stats(key.split('{')[0])
        
        # Get timer stats
        for key in self.metric_collector.timers.keys():
            name = key.split('{')[0]  # Remove labels for grouping
            metrics['timers'][name] = self.metric_collector.get_timer_stats(key.split('{')[0])
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall health
        health_status = "healthy"
        if any(alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] for alert in active_alerts):
            health_status = "unhealthy"
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            health_status = "degraded"
        
        return {
            'status': health_status,
            'timestamp': time.time(),
            'active_alerts': len(active_alerts),
            'monitoring_active': self._running,
            'alerts': [
                {
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in active_alerts
            ]
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

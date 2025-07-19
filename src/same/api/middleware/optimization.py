"""
Advanced API optimization middleware for high-performance request handling
"""

import logging
import asyncio
import time
import gzip
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import aioredis

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression types for response optimization"""
    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"


class CacheStrategy(Enum):
    """Caching strategies"""
    NO_CACHE = "no_cache"
    MEMORY_ONLY = "memory_only"
    REDIS_ONLY = "redis_only"
    HYBRID = "hybrid"


@dataclass
class MiddlewareConfig:
    """Configuration for optimization middleware"""
    enable_compression: bool = True
    compression_threshold: int = 1024  # Minimum bytes to compress
    enable_caching: bool = True
    cache_ttl: int = 300  # Cache TTL in seconds
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100  # Requests per minute
    enable_request_batching: bool = True
    batch_timeout: float = 0.1  # Batch timeout in seconds
    enable_metrics: bool = True


class RequestBatcher:
    """Batch similar requests for efficient processing"""
    
    def __init__(self, timeout: float = 0.1):
        self.timeout = timeout
        self.batches: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_futures: Dict[str, List[asyncio.Future]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
    async def add_request(self, batch_key: str, request_data: Dict[str, Any]) -> Any:
        """Add request to batch and return future result"""
        future = asyncio.Future()
        
        self.batches[batch_key].append(request_data)
        self.batch_futures[batch_key].append(future)
        
        # Start batch timer if not already running
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = asyncio.create_task(
                self._batch_timer(batch_key)
            )
        
        return await future
    
    async def _batch_timer(self, batch_key: str):
        """Timer for batch processing"""
        await asyncio.sleep(self.timeout)
        
        if batch_key in self.batches and self.batches[batch_key]:
            # Process batch
            batch_data = self.batches[batch_key]
            futures = self.batch_futures[batch_key]
            
            try:
                # Here you would call your batch processing function
                # For now, we'll just return the individual requests
                results = [{"processed": True, "data": data} for data in batch_data]
                
                # Resolve futures
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
                        
            except Exception as e:
                # Reject all futures with the error
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
            
            # Clean up
            del self.batches[batch_key]
            del self.batch_futures[batch_key]
            del self.batch_timers[batch_key]


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_second = requests_per_minute / 60.0
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": requests_per_minute, "last_update": time.time()}
        )
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = time.time()
        bucket = self.buckets[client_id]
        
        # Add tokens based on elapsed time
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(
            self.requests_per_minute,
            bucket["tokens"] + elapsed * self.tokens_per_second
        )
        bucket["last_update"] = now
        
        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False


class ResponseCompressor:
    """Response compression handler"""
    
    def __init__(self, threshold: int = 1024):
        self.threshold = threshold
    
    def should_compress(self, content: bytes, content_type: str) -> bool:
        """Check if content should be compressed"""
        if len(content) < self.threshold:
            return False
        
        # Compress text-based content types
        compressible_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/css",
            "text/javascript",
            "application/javascript"
        ]
        
        return any(ct in content_type for ct in compressible_types)
    
    def compress(self, content: bytes, compression_type: CompressionType) -> bytes:
        """Compress content using specified algorithm"""
        if compression_type == CompressionType.GZIP:
            return gzip.compress(content)
        elif compression_type == CompressionType.DEFLATE:
            import zlib
            return zlib.compress(content)
        elif compression_type == CompressionType.BROTLI:
            try:
                import brotli
                return brotli.compress(content)
            except ImportError:
                logger.warning("Brotli compression not available, falling back to gzip")
                return gzip.compress(content)
        
        return content


class APIOptimizationMiddleware(BaseHTTPMiddleware):
    """Advanced API optimization middleware"""
    
    def __init__(self, app, config: MiddlewareConfig = None):
        super().__init__(app)
        self.config = config or MiddlewareConfig()
        
        # Initialize components
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests) if self.config.enable_rate_limiting else None
        self.compressor = ResponseCompressor(self.config.compression_threshold) if self.config.enable_compression else None
        self.batcher = RequestBatcher(self.config.batch_timeout) if self.config.enable_request_batching else None
        
        # Cache storage
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'cached_responses': 0,
            'compressed_responses': 0,
            'rate_limited_requests': 0,
            'batched_requests': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0
        }
        
        logger.info("API optimization middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        try:
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Rate limiting
            if self.config.enable_rate_limiting and self.rate_limiter:
                if not self.rate_limiter.is_allowed(client_id):
                    self.metrics['rate_limited_requests'] += 1
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Rate limit exceeded"}
                    )
            
            # Check cache
            cache_key = None
            if self.config.enable_caching and request.method == "GET":
                cache_key = self._generate_cache_key(request)
                cached_response = await self._get_cached_response(cache_key)
                
                if cached_response:
                    self.metrics['cached_responses'] += 1
                    response = JSONResponse(content=cached_response['content'])
                    
                    # Add cache headers
                    response.headers["X-Cache"] = "HIT"
                    response.headers["X-Cache-Key"] = cache_key[:16]
                    
                    return response
            
            # Process request
            response = await call_next(request)
            
            # Cache response if applicable
            if cache_key and response.status_code == 200:
                await self._cache_response(cache_key, response)
            
            # Compress response if applicable
            if self.config.enable_compression and self.compressor:
                response = await self._compress_response(request, response)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{(time.time() - start_time) * 1000:.2f}ms"
            response.headers["X-Cache"] = "MISS" if cache_key else "DISABLED"
            
            # Update metrics
            self._update_metrics(start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in optimization middleware: {e}")
            # Return original response on error
            return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get from headers first
        client_id = request.headers.get("X-Client-ID")
        if client_id:
            return client_id
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        # Include URL, query parameters, and relevant headers
        key_data = {
            "url": str(request.url),
            "method": request.method,
            "headers": {
                k: v for k, v in request.headers.items()
                if k.lower() in ["accept", "accept-language", "authorization"]
            }
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        # Try memory cache first
        if cache_key in self.memory_cache:
            cached = self.memory_cache[cache_key]
            if time.time() - cached['timestamp'] < self.config.cache_ttl:
                return cached
            else:
                del self.memory_cache[cache_key]
        
        # Try Redis cache if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"api_cache:{cache_key}")
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.debug(f"Redis cache error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Response):
        """Cache response"""
        try:
            # Read response content
            if hasattr(response, 'body'):
                content = response.body
            else:
                content = b""
            
            # Try to parse as JSON
            try:
                json_content = json.loads(content.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return  # Don't cache non-JSON responses
            
            cached_data = {
                'content': json_content,
                'timestamp': time.time(),
                'headers': dict(response.headers)
            }
            
            # Store in memory cache
            self.memory_cache[cache_key] = cached_data
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        f"api_cache:{cache_key}",
                        self.config.cache_ttl,
                        json.dumps(cached_data)
                    )
                except Exception as e:
                    logger.debug(f"Redis cache store error: {e}")
            
            # Limit memory cache size
            if len(self.memory_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k]['timestamp']
                )[:100]
                for key in oldest_keys:
                    del self.memory_cache[key]
                    
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    async def _compress_response(self, request: Request, response: Response) -> Response:
        """Compress response if applicable"""
        try:
            # Check if client accepts compression
            accept_encoding = request.headers.get("accept-encoding", "")
            
            # Get response content
            if hasattr(response, 'body'):
                content = response.body
                content_type = response.headers.get("content-type", "")
                
                if self.compressor.should_compress(content, content_type):
                    # Choose compression method
                    if "br" in accept_encoding and CompressionType.BROTLI:
                        compressed_content = self.compressor.compress(content, CompressionType.BROTLI)
                        response.headers["content-encoding"] = "br"
                    elif "gzip" in accept_encoding:
                        compressed_content = self.compressor.compress(content, CompressionType.GZIP)
                        response.headers["content-encoding"] = "gzip"
                    elif "deflate" in accept_encoding:
                        compressed_content = self.compressor.compress(content, CompressionType.DEFLATE)
                        response.headers["content-encoding"] = "deflate"
                    else:
                        return response
                    
                    # Update response
                    response.body = compressed_content
                    response.headers["content-length"] = str(len(compressed_content))
                    response.headers["vary"] = "Accept-Encoding"
                    
                    self.metrics['compressed_responses'] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error compressing response: {e}")
            return response
    
    def _update_metrics(self, start_time: float):
        """Update performance metrics"""
        response_time = time.time() - start_time
        
        self.metrics['total_requests'] += 1
        self.metrics['total_response_time'] += response_time
        self.metrics['average_response_time'] = (
            self.metrics['total_response_time'] / self.metrics['total_requests']
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get middleware performance metrics"""
        return {
            **self.metrics,
            'cache_size': len(self.memory_cache),
            'config': {
                'compression_enabled': self.config.enable_compression,
                'caching_enabled': self.config.enable_caching,
                'rate_limiting_enabled': self.config.enable_rate_limiting,
                'batching_enabled': self.config.enable_request_batching
            }
        }
    
    async def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        
        if self.redis_client:
            try:
                # Clear Redis cache keys
                keys = await self.redis_client.keys("api_cache:*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
        
        logger.info("API cache cleared")


# Global middleware instance
api_optimization_middleware = APIOptimizationMiddleware

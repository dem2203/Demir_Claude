"""
DEMIR AI v8.0 - Redis Cache Client
HIGH-PERFORMANCE CACHING - ZERO MOCK DATA
ENTERPRISE GRADE CACHE MANAGEMENT
"""

import logging
import json
import pickle
import asyncio
import aioredis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
import time

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache management
    REAL-TIME CACHING - NO MOCK DATA
    """
    
    def __init__(self, config):
        self.config = config
        self.redis_url = config.database.redis_url
        self.default_ttl = config.database.redis_ttl
        self.max_connections = config.database.redis_max_connections
        
        self.redis = None
        self.pool = None
        self.is_connected = False
        
        # Cache namespaces
        self.namespaces = {
            'analysis': 'demir:analysis:',
            'signals': 'demir:signals:',
            'positions': 'demir:positions:',
            'orderbook': 'demir:orderbook:',
            'ticker': 'demir:ticker:',
            'candles': 'demir:candles:',
            'sentiment': 'demir:sentiment:',
            'technical': 'demir:technical:',
            'ml_predictions': 'demir:ml:',
            'user_data': 'demir:user:',
            'system': 'demir:system:',
            'temp': 'demir:temp:'
        }
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        logger.info(f"RedisCache initialized with URL: {self.redis_url}")
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.pool = await aioredis.create_redis_pool(
                self.redis_url,
                minsize=5,
                maxsize=self.max_connections,
                encoding='utf-8'
            )
            
            self.redis = self.pool
            self.is_connected = True
            
            # Test connection
            await self.redis.ping()
            
            # Get Redis info
            info = await self.redis.info()
            logger.info(f"Connected to Redis v{info.get('redis_version', 'unknown')}")
            logger.info(f"Redis memory usage: {info.get('used_memory_human', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.is_connected = False
            logger.info("Disconnected from Redis")
    
    async def close(self):
        """Alias for disconnect"""
        await self.disconnect()
    
    async def ping(self) -> bool:
        """Check Redis connection"""
        if not self.redis:
            return False
        
        try:
            response = await self.redis.ping()
            return response == b'PONG' or response == 'PONG'
        except:
            return False
    
    async def get(self, key: str, namespace: str = None) -> Optional[Any]:
        """
        Get value from cache
        REAL CACHE GET OPERATION
        """
        if not self.is_connected:
            return None
        
        self.total_requests += 1
        
        try:
            # Add namespace prefix if provided
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            # Get value
            value = await self.redis.get(key)
            
            if value is None:
                self.cache_misses += 1
                return None
            
            self.cache_hits += 1
            
            # Deserialize based on content
            return self._deserialize(value)
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None, namespace: str = None) -> bool:
        """
        Set value in cache
        REAL CACHE SET OPERATION
        """
        if not self.is_connected:
            return False
        
        try:
            # Add namespace prefix
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            # Serialize value
            serialized = self._serialize(value)
            
            # Set with TTL
            if ttl is None:
                ttl = self.default_ttl
            
            if ttl > 0:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str, namespace: str = None) -> bool:
        """Delete key from cache"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = None) -> bool:
        """Check if key exists"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            return await self.redis.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int, namespace: str = None) -> bool:
        """Set expiration for key"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            return await self.redis.expire(key, ttl)
            
        except Exception as e:
            logger.error(f"Redis expire error for key {key}: {e}")
            return False
    
    async def get_multiple(self, keys: List[str], namespace: str = None) -> Dict[str, Any]:
        """Get multiple values at once"""
        if not self.is_connected or not keys:
            return {}
        
        try:
            # Add namespace prefix
            if namespace and namespace in self.namespaces:
                keys = [self.namespaces[namespace] + k for k in keys]
            
            # Get all values
            values = await self.redis.mget(*keys)
            
            # Build result dict
            result = {}
            for i, key in enumerate(keys):
                if values[i] is not None:
                    # Remove namespace from key
                    clean_key = key
                    if namespace:
                        clean_key = key.replace(self.namespaces[namespace], '')
                    
                    result[clean_key] = self._deserialize(values[i])
            
            return result
            
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return {}
    
    async def set_multiple(self, data: Dict[str, Any], ttl: int = None, namespace: str = None) -> bool:
        """Set multiple values at once"""
        if not self.is_connected or not data:
            return False
        
        try:
            # Prepare data with namespace
            prepared = {}
            for key, value in data.items():
                full_key = key
                if namespace and namespace in self.namespaces:
                    full_key = self.namespaces[namespace] + key
                
                prepared[full_key] = self._serialize(value)
            
            # Set all values
            await self.redis.mset(prepared)
            
            # Set TTL if needed
            if ttl:
                for key in prepared.keys():
                    await self.redis.expire(key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, namespace: str = None) -> int:
        """Increment counter"""
        if not self.is_connected:
            return 0
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            return await self.redis.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Redis increment error for key {key}: {e}")
            return 0
    
    async def push(self, key: str, value: Any, namespace: str = None, max_length: int = 1000) -> bool:
        """Push to list (FIFO queue)"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            serialized = self._serialize(value)
            
            # Push to list
            await self.redis.lpush(key, serialized)
            
            # Trim to max length
            await self.redis.ltrim(key, 0, max_length - 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis push error for key {key}: {e}")
            return False
    
    async def pop(self, key: str, namespace: str = None) -> Optional[Any]:
        """Pop from list"""
        if not self.is_connected:
            return None
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            value = await self.redis.rpop(key)
            
            if value:
                return self._deserialize(value)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis pop error for key {key}: {e}")
            return None
    
    async def get_list(self, key: str, start: int = 0, end: int = -1, namespace: str = None) -> List[Any]:
        """Get list values"""
        if not self.is_connected:
            return []
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            values = await self.redis.lrange(key, start, end)
            
            return [self._deserialize(v) for v in values]
            
        except Exception as e:
            logger.error(f"Redis lrange error for key {key}: {e}")
            return []
    
    async def add_to_set(self, key: str, value: Any, namespace: str = None) -> bool:
        """Add to set"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            serialized = self._serialize(value)
            
            result = await self.redis.sadd(key, serialized)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis sadd error for key {key}: {e}")
            return False
    
    async def remove_from_set(self, key: str, value: Any, namespace: str = None) -> bool:
        """Remove from set"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            serialized = self._serialize(value)
            
            result = await self.redis.srem(key, serialized)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis srem error for key {key}: {e}")
            return False
    
    async def get_set_members(self, key: str, namespace: str = None) -> List[Any]:
        """Get all set members"""
        if not self.is_connected:
            return []
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            members = await self.redis.smembers(key)
            
            return [self._deserialize(m) for m in members]
            
        except Exception as e:
            logger.error(f"Redis smembers error for key {key}: {e}")
            return []
    
    async def set_hash(self, key: str, field: str, value: Any, namespace: str = None) -> bool:
        """Set hash field"""
        if not self.is_connected:
            return False
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            serialized = self._serialize(value)
            
            await self.redis.hset(key, field, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Redis hset error for key {key}: {e}")
            return False
    
    async def get_hash(self, key: str, field: str, namespace: str = None) -> Optional[Any]:
        """Get hash field"""
        if not self.is_connected:
            return None
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            value = await self.redis.hget(key, field)
            
            if value:
                return self._deserialize(value)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis hget error for key {key}: {e}")
            return None
    
    async def get_all_hash(self, key: str, namespace: str = None) -> Dict[str, Any]:
        """Get all hash fields"""
        if not self.is_connected:
            return {}
        
        try:
            if namespace and namespace in self.namespaces:
                key = self.namespaces[namespace] + key
            
            data = await self.redis.hgetall(key)
            
            result = {}
            for field, value in data.items():
                result[field] = self._deserialize(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Redis hgetall error for key {key}: {e}")
            return {}
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        if not self.is_connected:
            return 0
        
        try:
            serialized = self._serialize(message)
            return await self.redis.publish(channel, serialized)
            
        except Exception as e:
            logger.error(f"Redis publish error for channel {channel}: {e}")
            return 0
    
    async def subscribe(self, channels: List[str]) -> 'aioredis.Channel':
        """Subscribe to channels"""
        if not self.is_connected:
            return None
        
        try:
            channel = (await self.redis.subscribe(*channels))[0]
            return channel
            
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
            return None
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in namespace"""
        if not self.is_connected:
            return 0
        
        try:
            if namespace not in self.namespaces:
                logger.warning(f"Unknown namespace: {namespace}")
                return 0
            
            pattern = self.namespaces[namespace] + '*'
            
            # Get all keys
            keys = []
            cursor = b'0'
            
            while cursor:
                cursor, found_keys = await self.redis.scan(cursor, match=pattern)
                keys.extend(found_keys)
                
                if cursor == b'0':
                    break
            
            # Delete all keys
            if keys:
                return await self.redis.delete(*keys)
            
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear namespace error: {e}")
            return 0
    
    async def get_info(self) -> Dict:
        """Get Redis server info"""
        if not self.is_connected:
            return {}
        
        try:
            info = await self.redis.info()
            
            return {
                'version': info.get('redis_version'),
                'memory_used': info.get('used_memory_human'),
                'memory_peak': info.get('used_memory_peak_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'uptime_days': info.get('uptime_in_days')
            }
            
        except Exception as e:
            logger.error(f"Redis info error: {e}")
            return {}
    
    async def flush_db(self) -> bool:
        """Flush current database (DANGEROUS)"""
        if not self.is_connected:
            return False
        
        try:
            await self.redis.flushdb()
            logger.warning("Redis database flushed!")
            return True
            
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            return False
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        if value is None:
            return 'null'
        
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        
        try:
            return json.dumps(value)
        except:
            # Fall back to pickle for complex objects
            return pickle.dumps(value).hex()
    
    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """Deserialize value from storage"""
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        
        if value == 'null':
            return None
        
        try:
            return json.loads(value)
        except:
            try:
                # Try pickle
                return pickle.loads(bytes.fromhex(value))
            except:
                # Return as string if all else fails
                return value
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        hit_rate = 0
        if self.total_requests > 0:
            hit_rate = (self.cache_hits / self.total_requests) * 100
        
        return {
            'connected': self.is_connected,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    async def health_check(self) -> bool:
        """Check cache health"""
        return await self.ping()

"""
DEMIR AI v8.0 - Real-Time Data Pipeline Infrastructure
HIGH-PERFORMANCE DATA STREAMING AND PROCESSING
PROFESSIONAL ENTERPRISE IMPLEMENTATION
"""

import asyncio
import aioredis
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncpg
import motor.motor_asyncio
import websockets
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
from collections import deque
import struct
import zlib
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import uvloop

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


# ====================== DATA STRUCTURES ======================

@dataclass
class DataPoint:
    """Single data point in the pipeline"""
    timestamp: datetime
    source: str
    data_type: str
    payload: Dict[Any, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of data"""
        data_str = f"{self.timestamp}{self.source}{self.data_type}{self.payload}"
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def serialize(self) -> bytes:
        """Serialize to bytes for transmission"""
        return pickle.dumps(self)
    
    @staticmethod
    def deserialize(data: bytes) -> 'DataPoint':
        """Deserialize from bytes"""
        return pickle.loads(data)


@dataclass
class StreamConfig:
    """Configuration for data stream"""
    name: str
    source_type: str  # websocket, kafka, database, api
    connection_params: Dict[str, Any]
    buffer_size: int = 10000
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    compression: bool = True
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        'max_retries': 3,
        'backoff_factor': 2,
        'max_backoff': 60
    })


# ====================== WEBSOCKET STREAMER ======================

class WebSocketStreamer:
    """
    WebSocket Data Streamer
    REAL-TIME MARKET DATA STREAMING
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.url = config.connection_params['url']
        self.websocket = None
        self.running = False
        
        # Buffering
        self.buffer = deque(maxlen=config.buffer_size)
        self.batch_processor = None
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'reconnects': 0,
            'bytes_received': 0
        }
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        logger.info(f"WebSocketStreamer initialized for {self.url}")
    
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                max_size=10**8  # 100MB max message size
            )
            self.running = True
            logger.info(f"Connected to WebSocket: {self.url}")
            
            # Send subscription message if needed
            if 'subscribe' in self.config.connection_params:
                await self.websocket.send(
                    json.dumps(self.config.connection_params['subscribe'])
                )
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
    
    async def stream(self) -> AsyncIterator[DataPoint]:
        """
        Stream data from WebSocket
        ASYNC GENERATOR FOR REAL-TIME DATA
        """
        retry_count = 0
        backoff = 1
        
        while self.running:
            try:
                if not self.websocket:
                    await self.connect()
                
                async for message in self.websocket:
                    # Update statistics
                    self.stats['messages_received'] += 1
                    self.stats['bytes_received'] += len(message)
                    
                    # Parse message
                    try:
                        if isinstance(message, bytes):
                            data = json.loads(message.decode('utf-8'))
                        else:
                            data = json.loads(message)
                        
                        # Create DataPoint
                        data_point = DataPoint(
                            timestamp=datetime.now(),
                            source='websocket',
                            data_type=data.get('type', 'market_data'),
                            payload=data
                        )
                        
                        # Add to buffer
                        self.buffer.append(data_point)
                        
                        # Process callbacks
                        for callback in self.callbacks:
                            await callback(data_point)
                        
                        # Yield data point
                        yield data_point
                        
                        self.stats['messages_processed'] += 1
                        retry_count = 0  # Reset on success
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                        self.stats['errors'] += 1
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect...")
                self.stats['reconnects'] += 1
                
                # Exponential backoff
                if retry_count < self.config.retry_policy['max_retries']:
                    await asyncio.sleep(min(backoff, self.config.retry_policy['max_backoff']))
                    backoff *= self.config.retry_policy['backoff_factor']
                    retry_count += 1
                else:
                    logger.error("Max reconnection attempts reached")
                    self.running = False
                    break
                    
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
    
    def add_callback(self, callback: Callable):
        """Add callback for data processing"""
        self.callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """Get streaming statistics"""
        return self.stats


# ====================== KAFKA STREAMER ======================

class KafkaStreamer:
    """
    Kafka Data Streamer
    DISTRIBUTED STREAMING WITH KAFKA
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.bootstrap_servers = config.connection_params['bootstrap_servers']
        self.topic = config.connection_params['topic']
        self.group_id = config.connection_params.get('group_id', 'demirai_consumer')
        
        self.producer = None
        self.consumer = None
        self.running = False
        
        # Buffering
        self.send_buffer = deque(maxlen=config.buffer_size)
        self.receive_buffer = deque(maxlen=config.buffer_size)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0
        }
        
        logger.info(f"KafkaStreamer initialized for topic: {self.topic}")
    
    async def start_producer(self):
        """Start Kafka producer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip' if self.config.compression else None,
            acks='all',  # Wait for all replicas
            max_batch_size=16384,
            linger_ms=10
        )
        await self.producer.start()
        logger.info("Kafka producer started")
    
    async def start_consumer(self):
        """Start Kafka consumer"""
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            max_poll_records=500
        )
        await self.consumer.start()
        self.running = True
        logger.info(f"Kafka consumer started for topic: {self.topic}")
    
    async def send(self, data: Dict[str, Any]):
        """Send data to Kafka topic"""
        if not self.producer:
            await self.start_producer()
        
        try:
            # Add to send buffer
            self.send_buffer.append(data)
            
            # Send message
            result = await self.producer.send_and_wait(self.topic, data)
            
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += result.serialized_value_size
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
            self.stats['errors'] += 1
            raise
    
    async def send_batch(self, batch: List[Dict[str, Any]]):
        """Send batch of messages to Kafka"""
        if not self.producer:
            await self.start_producer()
        
        tasks = []
        for data in batch:
            task = self.producer.send(self.topic, data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        self.stats['messages_sent'] += success_count
        
        return results
    
    async def consume(self) -> AsyncIterator[DataPoint]:
        """
        Consume messages from Kafka
        ASYNC GENERATOR FOR STREAMING CONSUMPTION
        """
        if not self.consumer:
            await self.start_consumer()
        
        async for message in self.consumer:
            try:
                # Create DataPoint
                data_point = DataPoint(
                    timestamp=datetime.fromtimestamp(message.timestamp / 1000),
                    source='kafka',
                    data_type=message.value.get('type', 'stream_data'),
                    payload=message.value,
                    metadata={
                        'partition': message.partition,
                        'offset': message.offset,
                        'key': message.key
                    }
                )
                
                # Add to buffer
                self.receive_buffer.append(data_point)
                
                # Update stats
                self.stats['messages_received'] += 1
                self.stats['bytes_received'] += len(message.value)
                
                yield data_point
                
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
                self.stats['errors'] += 1
    
    async def stop(self):
        """Stop Kafka producer and consumer"""
        self.running = False
        
        if self.producer:
            await self.producer.stop()
            
        if self.consumer:
            await self.consumer.stop()
        
        logger.info("Kafka streamer stopped")


# ====================== DATABASE CONNECTOR ======================

class DatabaseConnector:
    """
    High-Performance Database Connector
    SUPPORTS POSTGRESQL, MONGODB, TIMESCALEDB
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_type = config['type']  # postgresql, mongodb, timescaledb
        self.connection_pool = None
        self.write_batch = []
        self.batch_size = config.get('batch_size', 1000)
        
        # Connection parameters
        self.connection_params = config['connection_params']
        
        # Statistics
        self.stats = {
            'queries_executed': 0,
            'rows_inserted': 0,
            'rows_updated': 0,
            'rows_deleted': 0,
            'errors': 0
        }
        
        logger.info(f"DatabaseConnector initialized for {self.db_type}")
    
    async def connect(self):
        """Establish database connection"""
        if self.db_type == 'postgresql':
            await self._connect_postgresql()
        elif self.db_type == 'mongodb':
            await self._connect_mongodb()
        elif self.db_type == 'timescaledb':
            await self._connect_timescaledb()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    async def _connect_postgresql(self):
        """Connect to PostgreSQL"""
        self.connection_pool = await asyncpg.create_pool(
            **self.connection_params,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool created")
    
    async def _connect_mongodb(self):
        """Connect to MongoDB"""
        client = motor.motor_asyncio.AsyncIOMotorClient(
            self.connection_params['url']
        )
        self.connection_pool = client[self.connection_params['database']]
        logger.info("MongoDB connection established")
    
    async def _connect_timescaledb(self):
        """Connect to TimescaleDB (PostgreSQL with time-series)"""
        await self._connect_postgresql()
        
        # Create hypertable if needed
        async with self.connection_pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS timescaledb;
            """)
            
            # Create main time-series table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    metadata JSONB
                );
            """)
            
            # Convert to hypertable
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_data', 'time', 
                                            chunk_time_interval => INTERVAL '1 day',
                                            if_not_exists => TRUE);
                """)
            except:
                pass  # Table might already be a hypertable
        
        logger.info("TimescaleDB initialized")
    
    async def insert_batch(self, table: str, data: List[Dict]):
        """
        Batch insert with high performance
        OPTIMIZED BULK INSERT
        """
        if self.db_type in ['postgresql', 'timescaledb']:
            await self._insert_batch_postgresql(table, data)
        elif self.db_type == 'mongodb':
            await self._insert_batch_mongodb(table, data)
    
    async def _insert_batch_postgresql(self, table: str, data: List[Dict]):
        """PostgreSQL batch insert using COPY"""
        if not data:
            return
        
        async with self.connection_pool.acquire() as conn:
            # Prepare data for COPY
            columns = list(data[0].keys())
            
            # Use COPY for maximum performance
            result = await conn.copy_records_to_table(
                table,
                records=[tuple(row[col] for col in columns) for row in data],
                columns=columns
            )
            
            self.stats['rows_inserted'] += len(data)
            self.stats['queries_executed'] += 1
            
            logger.info(f"Inserted {len(data)} rows into {table}")
    
    async def _insert_batch_mongodb(self, collection_name: str, data: List[Dict]):
        """MongoDB batch insert"""
        if not data:
            return
        
        collection = self.connection_pool[collection_name]
        result = await collection.insert_many(data)
        
        self.stats['rows_inserted'] += len(result.inserted_ids)
        self.stats['queries_executed'] += 1
        
        logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
    
    async def query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute query and return results"""
        if self.db_type in ['postgresql', 'timescaledb']:
            return await self._query_postgresql(query, params)
        elif self.db_type == 'mongodb':
            raise NotImplementedError("Use MongoDB-specific query methods")
    
    async def _query_postgresql(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute PostgreSQL query"""
        async with self.connection_pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
            
            self.stats['queries_executed'] += 1
            
            # Convert to dict
            return [dict(row) for row in rows]
    
    async def stream_query(self, query: str, params: Optional[tuple] = None) -> AsyncIterator[Dict]:
        """
        Stream query results
        MEMORY-EFFICIENT LARGE RESULT SETS
        """
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                cursor = await conn.cursor(query, *params) if params else await conn.cursor(query)
                
                async for row in cursor:
                    yield dict(row)
    
    async def close(self):
        """Close database connections"""
        if self.db_type in ['postgresql', 'timescaledb']:
            await self.connection_pool.close()
        elif self.db_type == 'mongodb':
            self.connection_pool.client.close()
        
        logger.info("Database connection closed")


# ====================== REDIS CACHE ======================

class RedisCache:
    """
    High-Performance Redis Cache
    IN-MEMORY DATA STORAGE AND PUBSUB
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_url = config['url']
        self.redis_client = None
        self.pubsub = None
        self.subscriptions = {}
        
        # Cache configuration
        self.default_ttl = config.get('default_ttl', 3600)  # 1 hour
        self.max_memory = config.get('max_memory', '1gb')
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'pubsub_messages': 0
        }
        
        logger.info("RedisCache initialized")
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = await aioredis.create_redis_pool(
            self.redis_url,
            minsize=5,
            maxsize=10,
            encoding='utf-8'
        )
        
        # Configure Redis
        await self.redis_client.config_set('maxmemory', self.max_memory)
        await self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')
        
        logger.info("Redis connection established")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        value = await self.redis_client.get(key)
        
        if value:
            self.stats['hits'] += 1
            # Deserialize if needed
            try:
                return json.loads(value)
            except:
                return value
        else:
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        # Serialize if needed
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        ttl = ttl or self.default_ttl
        await self.redis_client.setex(key, ttl, value)
        self.stats['sets'] += 1
    
    async def delete(self, key: str):
        """Delete key from cache"""
        await self.redis_client.delete(key)
        self.stats['deletes'] += 1
    
    async def get_batch(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        values = await self.redis_client.mget(*keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value:
                self.stats['hits'] += 1
                try:
                    result[key] = json.loads(value)
                except:
                    result[key] = value
            else:
                self.stats['misses'] += 1
        
        return result
    
    async def set_batch(self, data: Dict[str, Any], ttl: Optional[int] = None):
        """Set multiple values"""
        ttl = ttl or self.default_ttl
        
        pipe = self.redis_client.pipeline()
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            pipe.setex(key, ttl, value)
        
        await pipe.execute()
        self.stats['sets'] += len(data)
    
    async def publish(self, channel: str, message: Any):
        """Publish message to channel"""
        if isinstance(message, (dict, list)):
            message = json.dumps(message)
        
        await self.redis_client.publish(channel, message)
        self.stats['pubsub_messages'] += 1
    
    async def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel with callback"""
        if channel not in self.subscriptions:
            ch = (await self.redis_client.subscribe(channel))[0]
            self.subscriptions[channel] = {
                'channel': ch,
                'callbacks': [callback]
            }
            
            # Start listening
            asyncio.create_task(self._listen_channel(channel))
        else:
            self.subscriptions[channel]['callbacks'].append(callback)
    
    async def _listen_channel(self, channel: str):
        """Listen to channel and execute callbacks"""
        ch = self.subscriptions[channel]['channel']
        
        async for message in ch.iter():
            self.stats['pubsub_messages'] += 1
            
            # Parse message
            try:
                data = json.loads(message)
            except:
                data = message
            
            # Execute callbacks
            for callback in self.subscriptions[channel]['callbacks']:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        logger.info("Redis connection closed")


# ====================== DATA PROCESSOR ======================

class DataProcessor:
    """
    High-Performance Data Processing Engine
    TRANSFORMS AND ENRICHES DATA IN REAL-TIME
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.processors: List[Callable] = []
        self.enrichers: List[Callable] = []
        
        # Processing configuration
        self.batch_size = config.get('batch_size', 100)
        self.parallel_workers = config.get('parallel_workers', 4)
        
        # Thread/Process pools for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.parallel_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.parallel_workers)
        
        # Statistics
        self.stats = {
            'processed': 0,
            'enriched': 0,
            'errors': 0,
            'processing_time': 0
        }
        
        logger.info(f"DataProcessor initialized with {self.parallel_workers} workers")
    
    def add_processor(self, processor: Callable):
        """Add data processor function"""
        self.processors.append(processor)
    
    def add_enricher(self, enricher: Callable):
        """Add data enricher function"""
        self.enrichers.append(enricher)
    
    async def process(self, data: DataPoint) -> DataPoint:
        """
        Process single data point
        APPLIES ALL PROCESSORS AND ENRICHERS
        """
        start_time = datetime.now()
        
        try:
            # Apply processors
            for processor in self.processors:
                if asyncio.iscoroutinefunction(processor):
                    data = await processor(data)
                else:
                    # Run CPU-bound processor in thread pool
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(self.thread_pool, processor, data)
            
            # Apply enrichers
            for enricher in self.enrichers:
                if asyncio.iscoroutinefunction(enricher):
                    data = await enricher(data)
                else:
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(self.thread_pool, enricher, data)
            
            self.stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.stats['errors'] += 1
            raise
        
        finally:
            self.stats['processing_time'] += (datetime.now() - start_time).total_seconds()
        
        return data
    
    async def process_batch(self, batch: List[DataPoint]) -> List[DataPoint]:
        """
        Process batch of data points
        PARALLEL BATCH PROCESSING
        """
        # Process in parallel
        tasks = [self.process(data) for data in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        processed = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                self.stats['errors'] += 1
            else:
                processed.append(result)
        
        return processed
    
    async def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform pandas DataFrame
        VECTORIZED OPERATIONS FOR PERFORMANCE
        """
        start_time = datetime.now()
        
        # Apply transformations
        for processor in self.processors:
            if hasattr(processor, '__name__') and 'dataframe' in processor.__name__:
                df = processor(df)
        
        self.stats['processing_time'] += (datetime.now() - start_time).total_seconds()
        
        return df
    
    def shutdown(self):
        """Shutdown processor pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("DataProcessor shutdown complete")


# ====================== DATA PIPELINE ORCHESTRATOR ======================

class DataPipelineOrchestrator:
    """
    Master Data Pipeline Orchestrator
    COORDINATES ALL DATA FLOW COMPONENTS
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Pipeline components
        self.streamers: Dict[str, Any] = {}
        self.database = None
        self.cache = None
        self.processor = None
        
        # Pipeline state
        self.running = False
        self.tasks = []
        
        # Monitoring
        self.metrics = {
            'total_messages': 0,
            'messages_per_second': 0,
            'pipeline_latency': 0,
            'error_rate': 0
        }
        
        # Message queue for pipeline
        self.message_queue = asyncio.Queue(maxsize=10000)
        
        logger.info("DataPipelineOrchestrator initialized")
    
    async def initialize(self):
        """Initialize all pipeline components"""
        # Initialize database
        if 'database' in self.config:
            self.database = DatabaseConnector(self.config['database'])
            await self.database.connect()
        
        # Initialize cache
        if 'redis' in self.config:
            self.cache = RedisCache(self.config['redis'])
            await self.cache.connect()
        
        # Initialize processor
        if 'processor' in self.config:
            self.processor = DataProcessor(self.config['processor'])
        
        # Initialize streamers
        for name, stream_config in self.config.get('streams', {}).items():
            if stream_config['source_type'] == 'websocket':
                self.streamers[name] = WebSocketStreamer(StreamConfig(**stream_config))
            elif stream_config['source_type'] == 'kafka':
                self.streamers[name] = KafkaStreamer(StreamConfig(**stream_config))
        
        logger.info("Pipeline components initialized")
    
    async def start(self):
        """Start the data pipeline"""
        self.running = True
        
        # Start streaming tasks
        for name, streamer in self.streamers.items():
            if isinstance(streamer, WebSocketStreamer):
                task = asyncio.create_task(self._process_websocket_stream(name, streamer))
            elif isinstance(streamer, KafkaStreamer):
                task = asyncio.create_task(self._process_kafka_stream(name, streamer))
            
            self.tasks.append(task)
        
        # Start pipeline processor
        processor_task = asyncio.create_task(self._process_pipeline())
        self.tasks.append(processor_task)
        
        # Start metrics collector
        metrics_task = asyncio.create_task(self._collect_metrics())
        self.tasks.append(metrics_task)
        
        logger.info("Data pipeline started")
    
    async def _process_websocket_stream(self, name: str, streamer: WebSocketStreamer):
        """Process WebSocket stream"""
        logger.info(f"Starting WebSocket stream: {name}")
        
        async for data_point in streamer.stream():
            if not self.running:
                break
            
            # Add to pipeline queue
            await self.message_queue.put(data_point)
            self.metrics['total_messages'] += 1
    
    async def _process_kafka_stream(self, name: str, streamer: KafkaStreamer):
        """Process Kafka stream"""
        logger.info(f"Starting Kafka stream: {name}")
        
        async for data_point in streamer.consume():
            if not self.running:
                break
            
            # Add to pipeline queue
            await self.message_queue.put(data_point)
            self.metrics['total_messages'] += 1
    
    async def _process_pipeline(self):
        """Main pipeline processing loop"""
        batch = []
        last_batch_time = datetime.now()
        
        while self.running:
            try:
                # Get message with timeout
                try:
                    data_point = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                    batch.append(data_point)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if ready
                should_process = (
                    len(batch) >= self.config.get('batch_size', 100) or
                    (datetime.now() - last_batch_time).total_seconds() > 1.0
                )
                
                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = datetime.now()
                    
            except Exception as e:
                logger.error(f"Pipeline processing error: {e}")
                self.metrics['error_rate'] += 1
    
    async def _process_batch(self, batch: List[DataPoint]):
        """Process a batch of data points"""
        start_time = datetime.now()
        
        # Process data
        if self.processor:
            batch = await self.processor.process_batch(batch)
        
        # Cache data
        if self.cache:
            cache_data = {}
            for data_point in batch:
                key = f"{data_point.source}:{data_point.timestamp.timestamp()}"
                cache_data[key] = data_point.payload
            
            await self.cache.set_batch(cache_data)
        
        # Store in database
        if self.database:
            db_data = [
                {
                    'timestamp': dp.timestamp,
                    'source': dp.source,
                    'data_type': dp.data_type,
                    'payload': json.dumps(dp.payload),
                    'checksum': dp.checksum
                }
                for dp in batch
            ]
            
            await self.database.insert_batch('data_points', db_data)
        
        # Update metrics
        self.metrics['pipeline_latency'] = (datetime.now() - start_time).total_seconds()
    
    async def _collect_metrics(self):
        """Collect pipeline metrics"""
        last_count = 0
        
        while self.running:
            await asyncio.sleep(1)
            
            # Calculate messages per second
            current_count = self.metrics['total_messages']
            self.metrics['messages_per_second'] = current_count - last_count
            last_count = current_count
            
            # Log metrics periodically
            if current_count % 1000 == 0:
                logger.info(f"Pipeline metrics: {self.metrics}")
    
    async def stop(self):
        """Stop the data pipeline"""
        logger.info("Stopping data pipeline...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close connections
        if self.database:
            await self.database.close()
        
        if self.cache:
            await self.cache.close()
        
        if self.processor:
            self.processor.shutdown()
        
        for streamer in self.streamers.values():
            if hasattr(streamer, 'stop'):
                await streamer.stop()
            elif hasattr(streamer, 'disconnect'):
                await streamer.disconnect()
        
        logger.info("Data pipeline stopped")
    
    def get_metrics(self) -> Dict:
        """Get pipeline metrics"""
        metrics = self.metrics.copy()
        
        # Add component metrics
        if self.database:
            metrics['database'] = self.database.stats
        
        if self.cache:
            metrics['cache'] = self.cache.stats
        
        if self.processor:
            metrics['processor'] = self.processor.stats
        
        for name, streamer in self.streamers.items():
            if hasattr(streamer, 'get_stats'):
                metrics[f'stream_{name}'] = streamer.get_stats()
        
        return metrics


# ====================== EXAMPLE USAGE ======================

async def example_pipeline():
    """Example data pipeline setup"""
    
    config = {
        'database': {
            'type': 'timescaledb',
            'connection_params': {
                'host': 'localhost',
                'port': 5432,
                'database': 'demirai',
                'user': 'postgres',
                'password': 'password'
            }
        },
        'redis': {
            'url': 'redis://localhost:6379',
            'default_ttl': 3600
        },
        'processor': {
            'batch_size': 100,
            'parallel_workers': 4
        },
        'streams': {
            'binance': {
                'source_type': 'websocket',
                'connection_params': {
                    'url': 'wss://stream.binance.com:9443/ws/btcusdt@trade',
                    'subscribe': {
                        'method': 'SUBSCRIBE',
                        'params': ['btcusdt@trade', 'btcusdt@depth'],
                        'id': 1
                    }
                },
                'buffer_size': 10000
            }
        }
    }
    
    # Create and start pipeline
    pipeline = DataPipelineOrchestrator(config)
    await pipeline.initialize()
    await pipeline.start()
    
    # Run for some time
    await asyncio.sleep(60)
    
    # Get metrics
    metrics = pipeline.get_metrics()
    print(f"Pipeline metrics: {metrics}")
    
    # Stop pipeline
    await pipeline.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_pipeline())

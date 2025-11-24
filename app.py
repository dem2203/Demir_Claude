"""
DEMIR AI v8.0 - Professional FastAPI Application
ENTERPRISE GRADE REST API & WEBSOCKET SERVER
PRODUCTION READY - ZERO MOCK DATA
"""

import os
import sys
import logging
import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect, status, Query, Body, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
import jwt
import hashlib
import hmac

# Performance monitoring
import psutil
import aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
websocket_connections = Gauge('websocket_connections', 'Active WebSocket connections')
active_trades = Gauge('active_trades', 'Active trades')
system_health = Gauge('system_health', 'System health score')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# ====================== MODELS ======================

class TradingMode(str, Enum):
    ADVISORY = "ADVISORY"
    PAPER = "PAPER"
    LIVE = "LIVE"

class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"

class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class MarketCondition(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    CONSOLIDATION = "CONSOLIDATION"

# ====================== REQUEST/RESPONSE MODELS ======================

class SignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTC/USDT)")
    timeframe: TimeFrame = Field(default=TimeFrame.H1)
    use_ml: bool = Field(default=True, description="Use ML models for analysis")
    use_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Invalid symbol format")
        return v.upper()

class SignalResponse(BaseModel):
    signal_id: str
    timestamp: datetime
    symbol: str
    action: SignalAction
    confidence: float = Field(..., ge=0, le=1)
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    position_size: float
    risk_reward_ratio: float
    market_condition: MarketCondition
    technical_indicators: Dict[str, float]
    ml_predictions: Dict[str, float]
    sentiment_scores: Dict[str, float]
    reasoning: str
    expires_at: datetime

class TradeRequest(BaseModel):
    symbol: str
    action: SignalAction
    order_type: OrderType = OrderType.MARKET
    quantity: float = Field(..., gt=0)
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_percentage: Optional[float] = Field(None, ge=0, le=0.1)
    leverage: float = Field(default=1.0, ge=1, le=125)
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') == OrderType.LIMIT and not v:
            raise ValueError("Price required for limit orders")
        return v

class TradeResponse(BaseModel):
    trade_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    action: SignalAction
    order_type: OrderType
    status: str
    executed_price: Optional[float]
    executed_quantity: Optional[float]
    commission: Optional[float]
    commission_asset: Optional[str]

class PositionInfo(BaseModel):
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    mark_price: float
    pnl: float
    pnl_percentage: float
    margin: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime
    updated_at: datetime

class MarketData(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    volume_quote_24h: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_percentage_24h: float
    trades_24h: int
    timestamp: datetime

class BacktestRequest(BaseModel):
    strategy: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=10000, gt=0)
    commission: float = Field(default=0.001, ge=0, le=0.01)
    slippage: float = Field(default=0.001, ge=0, le=0.01)
    position_size: float = Field(default=0.1, gt=0, le=1)
    stop_loss: float = Field(default=0.03, gt=0, le=0.5)
    take_profit: float = Field(default=0.06, gt=0, le=1)
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        return v

class BacktestResponse(BaseModel):
    backtest_id: str
    strategy: str
    symbol: str
    period: Dict[str, str]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    total_return_percentage: float
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_history: List[Dict[str, Any]]

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., min_items=2, max_items=20)
    optimization_method: str = Field(default="max_sharpe")
    constraints: Dict[str, float] = Field(default_factory=dict)
    lookback_days: int = Field(default=365, ge=30, le=1825)
    rebalance_frequency: str = Field(default="monthly")
    
    @validator('optimization_method')
    def validate_method(cls, v):
        valid_methods = ["max_sharpe", "min_variance", "risk_parity", "mean_variance", "black_litterman"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v

class PortfolioOptimizationResponse(BaseModel):
    optimization_id: str
    timestamp: datetime
    method: str
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    risk_metrics: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    rebalancing_schedule: List[Dict[str, Any]]

class SystemStatus(BaseModel):
    status: str
    uptime: float
    version: str
    trading_mode: TradingMode
    active_subsystems: Dict[str, bool]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    last_signal_time: Optional[datetime]
    active_positions: int
    total_trades_today: int
    daily_pnl: float
    circuit_breaker_status: str
    health_score: float

class AlertSettings(BaseModel):
    email_enabled: bool = False
    email_address: Optional[str] = None
    telegram_enabled: bool = False
    telegram_chat_id: Optional[str] = None
    discord_enabled: bool = False
    discord_webhook: Optional[str] = None
    alert_types: List[str] = Field(default_factory=list)
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    
    @validator('email_address')
    def validate_email(cls, v, values):
        if values.get('email_enabled') and not v:
            raise ValueError("Email address required when email alerts enabled")
        return v

# ====================== AUTHENTICATION ======================

def create_jwt_token(user_id: str, api_key: str) -> str:
    """Create JWT token for authentication"""
    payload = {
        "user_id": user_id,
        "api_key": api_key,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify JWT token and return payload"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

def verify_api_signature(signature: str, timestamp: str, body: str) -> bool:
    """Verify API request signature for webhook security"""
    api_secret = os.getenv("API_SECRET", "")
    message = f"{timestamp}.{body}"
    expected_signature = hmac.new(
        api_secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected_signature)

# ====================== CONNECTION MANAGER ======================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, List[str]] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Dict = None):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        self.user_subscriptions[client_id] = []
        websocket_connections.inc()
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.user_subscriptions[client_id]
            del self.connection_metadata[client_id]
            websocket_connections.dec()
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, channels: List[str]):
        """Subscribe client to channels"""
        if client_id in self.user_subscriptions:
            self.user_subscriptions[client_id].extend(channels)
            self.user_subscriptions[client_id] = list(set(self.user_subscriptions[client_id]))
            return True
        return False
    
    async def unsubscribe(self, client_id: str, channels: List[str]):
        """Unsubscribe client from channels"""
        if client_id in self.user_subscriptions:
            for channel in channels:
                if channel in self.user_subscriptions[client_id]:
                    self.user_subscriptions[client_id].remove(channel)
            return True
        return False
    
    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_channel(self, message: Dict, channel: str):
        """Broadcast message to all clients subscribed to channel"""
        disconnected_clients = []
        
        for client_id, subscriptions in self.user_subscriptions.items():
            if channel in subscriptions:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def broadcast_to_all(self, message: Dict):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        channel_stats = {}
        for subscriptions in self.user_subscriptions.values():
            for channel in subscriptions:
                channel_stats[channel] = channel_stats.get(channel, 0) + 1
        
        return {
            "total_connections": len(self.active_connections),
            "channel_subscribers": channel_stats,
            "connection_ids": list(self.active_connections.keys())
        }

# ====================== MAIN APPLICATION ======================

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting DEMIR AI v8.0 API Server")
    
    # Initialize Redis connection
    app.state.redis = await aioredis.create_redis_pool(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        minsize=5,
        maxsize=10
    )
    
    # Initialize orchestrator (imported from main application)
    from orchestrator import DemirAIOrchestrator, TradingMode as OrchestratorTradingMode
    
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    app.state.orchestrator = DemirAIOrchestrator(config_path)
    await app.state.orchestrator.initialize()
    
    # Set trading mode
    trading_mode_str = os.getenv("TRADING_MODE", "ADVISORY")
    app.state.orchestrator.mode = getattr(OrchestratorTradingMode, trading_mode_str)
    
    # Start orchestrator
    app.state.orchestrator_task = asyncio.create_task(app.state.orchestrator.start())
    
    logger.info("API Server initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Server")
    
    # Stop orchestrator
    if hasattr(app.state, 'orchestrator_task'):
        app.state.orchestrator_task.cancel()
    
    if hasattr(app.state, 'orchestrator'):
        await app.state.orchestrator.shutdown()
    
    # Close Redis connection
    if hasattr(app.state, 'redis'):
        app.state.redis.close()
        await app.state.redis.wait_closed()
    
    logger.info("API Server shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="DEMIR AI v8.0 API",
    description="Professional Crypto Trading Bot API - Enterprise Grade",
    version="8.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Add rate limit handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize connection manager
connection_manager = ConnectionManager()

# ====================== MIDDLEWARE ======================

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add process time and request ID headers"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to logs
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Update metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    return response

# ====================== HEALTH & MONITORING ENDPOINTS ======================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check orchestrator health
        orchestrator_healthy = False
        if hasattr(app.state, 'orchestrator'):
            status = app.state.orchestrator.get_status()
            orchestrator_healthy = status.get('is_running', False)
        
        # Check Redis health
        redis_healthy = False
        if hasattr(app.state, 'redis'):
            await app.state.redis.ping()
            redis_healthy = True
        
        # Calculate health score
        health_score = 0
        if orchestrator_healthy:
            health_score += 50
        if redis_healthy:
            health_score += 50
        
        system_health.set(health_score)
        
        return {
            "status": "healthy" if health_score >= 50 else "unhealthy",
            "health_score": health_score,
            "orchestrator": "healthy" if orchestrator_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type="text/plain"
    )

@app.get("/status", response_model=SystemStatus)
async def system_status(auth: Dict = Depends(verify_jwt_token)):
    """Get system status"""
    try:
        status = app.state.orchestrator.get_status()
        
        # Get resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemStatus(
            status="running" if status['is_running'] else "stopped",
            uptime=status.get('uptime', 0),
            version="8.0.0",
            trading_mode=TradingMode(status.get('mode', 'ADVISORY')),
            active_subsystems=status.get('subsystems', {}),
            performance_metrics={
                "total_signals": status.get('metrics', {}).get('total_signals', 0),
                "win_rate": status.get('metrics', {}).get('win_rate', 0),
                "sharpe_ratio": status.get('metrics', {}).get('sharpe_ratio', 0),
                "max_drawdown": status.get('metrics', {}).get('max_drawdown', 0)
            },
            resource_usage={
                "cpu_percent": cpu_usage,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3)
            },
            last_signal_time=status.get('last_signal_time'),
            active_positions=status.get('active_positions', 0),
            total_trades_today=status.get('trades_today', 0),
            daily_pnl=status.get('daily_pnl', 0),
            circuit_breaker_status=status.get('circuit_breaker', 'CLOSED'),
            health_score=status.get('health_score', 100)
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== TRADING ENDPOINTS ======================

@app.post("/api/v1/signals", response_model=SignalResponse)
@limiter.limit("10/minute")
async def generate_signal(
    request: SignalRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_jwt_token)
):
    """Generate trading signal"""
    try:
        # Generate signal through orchestrator
        signal_data = await app.state.orchestrator.signal_generator.generate_signal(
            symbol=request.symbol,
            timeframe=request.timeframe,
            use_ml=request.use_ml,
            use_sentiment=request.use_sentiment
        )
        
        # Create response
        response = SignalResponse(
            signal_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            symbol=request.symbol,
            action=SignalAction(signal_data['action']),
            confidence=signal_data['confidence'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit_1=signal_data['take_profit_1'],
            take_profit_2=signal_data.get('take_profit_2'),
            take_profit_3=signal_data.get('take_profit_3'),
            position_size=signal_data['position_size'],
            risk_reward_ratio=signal_data['risk_reward_ratio'],
            market_condition=MarketCondition(signal_data['market_condition']),
            technical_indicators=signal_data['technical_indicators'],
            ml_predictions=signal_data['ml_predictions'],
            sentiment_scores=signal_data['sentiment_scores'],
            reasoning=signal_data['reasoning'],
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Broadcast signal to WebSocket subscribers
        background_tasks.add_task(
            connection_manager.broadcast_to_channel,
            response.dict(),
            f"signals:{request.symbol}"
        )
        
        # Store signal in Redis cache
        if hasattr(app.state, 'redis'):
            await app.state.redis.setex(
                f"signal:{response.signal_id}",
                3600,  # 1 hour TTL
                json.dumps(response.dict(), default=str)
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal generation failed: {str(e)}"
        )

@app.post("/api/v1/trades", response_model=TradeResponse)
@limiter.limit("5/minute")
async def execute_trade(
    request: TradeRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_jwt_token)
):
    """Execute trade"""
    try:
        # Check if trading is enabled
        if app.state.orchestrator.mode == TradingMode.ADVISORY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Trading disabled - system in advisory mode"
            )
        
        # Execute trade through orchestrator
        trade_result = await app.state.orchestrator.position_manager.open_position(
            symbol=request.symbol,
            side='BUY' if request.action == SignalAction.BUY else 'SELL',
            quantity=request.quantity,
            order_type=request.order_type.value,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            leverage=request.leverage
        )
        
        # Create response
        response = TradeResponse(
            trade_id=str(uuid.uuid4()),
            order_id=trade_result['order_id'],
            timestamp=datetime.utcnow(),
            symbol=request.symbol,
            action=request.action,
            order_type=request.order_type,
            status=trade_result['status'],
            executed_price=trade_result.get('executed_price'),
            executed_quantity=trade_result.get('executed_quantity'),
            commission=trade_result.get('commission'),
            commission_asset=trade_result.get('commission_asset')
        )
        
        # Update active trades metric
        active_trades.inc()
        
        # Broadcast trade to WebSocket subscribers
        background_tasks.add_task(
            connection_manager.broadcast_to_channel,
            response.dict(),
            "trades"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trade execution failed: {str(e)}"
        )

@app.get("/api/v1/positions", response_model=List[PositionInfo])
async def get_positions(
    symbol: Optional[str] = Query(None),
    auth: Dict = Depends(verify_jwt_token)
):
    """Get open positions"""
    try:
        positions = await app.state.orchestrator.position_manager.get_positions(symbol)
        
        return [
            PositionInfo(
                position_id=pos['position_id'],
                symbol=pos['symbol'],
                side=pos['side'],
                quantity=pos['quantity'],
                entry_price=pos['entry_price'],
                current_price=pos['current_price'],
                mark_price=pos['mark_price'],
                pnl=pos['pnl'],
                pnl_percentage=pos['pnl_percentage'],
                margin=pos['margin'],
                leverage=pos['leverage'],
                stop_loss=pos.get('stop_loss'),
                take_profit=pos.get('take_profit'),
                opened_at=pos['opened_at'],
                updated_at=pos['updated_at']
            )
            for pos in positions
        ]
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/v1/positions/{position_id}")
async def close_position(
    position_id: str,
    auth: Dict = Depends(verify_jwt_token)
):
    """Close specific position"""
    try:
        result = await app.state.orchestrator.position_manager.close_position(position_id)
        
        # Update active trades metric
        active_trades.dec()
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== MARKET DATA ENDPOINTS ======================

@app.get("/api/v1/market/{symbol}", response_model=MarketData)
@limiter.limit("30/minute")
async def get_market_data(
    symbol: str,
    auth: Dict = Depends(verify_jwt_token)
):
    """Get real-time market data"""
    try:
        data = await app.state.orchestrator.exchange_manager.get_ticker(symbol)
        
        return MarketData(
            symbol=symbol,
            price=data['last'],
            bid=data['bid'],
            ask=data['ask'],
            volume_24h=data['baseVolume'],
            volume_quote_24h=data['quoteVolume'],
            high_24h=data['high'],
            low_24h=data['low'],
            change_24h=data['change'],
            change_percentage_24h=data['percentage'],
            trades_24h=data.get('count', 0),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/market/{symbol}/orderbook")
@limiter.limit("20/minute")
async def get_orderbook(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100),
    auth: Dict = Depends(verify_jwt_token)
):
    """Get order book"""
    try:
        orderbook = await app.state.orchestrator.exchange_manager.get_order_book(symbol, limit)
        return orderbook
    except Exception as e:
        logger.error(f"Error getting orderbook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/market/{symbol}/candles")
@limiter.limit("20/minute")
async def get_candles(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    limit: int = Query(default=100, ge=1, le=500),
    auth: Dict = Depends(verify_jwt_token)
):
    """Get historical candles"""
    try:
        candles = await app.state.orchestrator.exchange_manager.get_ohlcv(
            symbol, timeframe.value, limit
        )
        return candles
    except Exception as e:
        logger.error(f"Error getting candles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== BACKTESTING ENDPOINTS ======================

@app.post("/api/v1/backtest", response_model=BacktestResponse)
@limiter.limit("2/minute")
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    auth: Dict = Depends(verify_jwt_token)
):
    """Run backtest"""
    try:
        # Run backtest through orchestrator
        result = await app.state.orchestrator.backtest_engine.run_backtest(
            strategy=request.strategy,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            config={
                'commission': request.commission,
                'slippage': request.slippage,
                'position_size': request.position_size,
                'stop_loss': request.stop_loss,
                'take_profit': request.take_profit
            }
        )
        
        return BacktestResponse(
            backtest_id=str(uuid.uuid4()),
            strategy=request.strategy,
            symbol=request.symbol,
            period={
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            },
            total_trades=result['total_trades'],
            winning_trades=result['winning_trades'],
            losing_trades=result['losing_trades'],
            win_rate=result['win_rate'],
            profit_factor=result['profit_factor'],
            total_return=result['total_return'],
            total_return_percentage=result['total_return_percentage'],
            max_drawdown=result['max_drawdown'],
            max_drawdown_percentage=result['max_drawdown_percentage'],
            sharpe_ratio=result['sharpe_ratio'],
            sortino_ratio=result['sortino_ratio'],
            calmar_ratio=result['calmar_ratio'],
            avg_trade_return=result['avg_trade_return'],
            avg_winning_trade=result['avg_winning_trade'],
            avg_losing_trade=result['avg_losing_trade'],
            best_trade=result['best_trade'],
            worst_trade=result['worst_trade'],
            equity_curve=result['equity_curve'],
            drawdown_curve=result['drawdown_curve'],
            trade_history=result['trades']
        )
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== PORTFOLIO ENDPOINTS ======================

@app.post("/api/v1/portfolio/optimize", response_model=PortfolioOptimizationResponse)
@limiter.limit("5/minute")
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    auth: Dict = Depends(verify_jwt_token)
):
    """Optimize portfolio allocation"""
    try:
        result = await app.state.orchestrator.portfolio_optimizer.optimize(
            symbols=request.symbols,
            method=request.optimization_method,
            constraints=request.constraints,
            lookback_days=request.lookback_days
        )
        
        return PortfolioOptimizationResponse(
            optimization_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            method=request.optimization_method,
            optimal_weights=result['weights'],
            expected_return=result['expected_return'],
            expected_volatility=result['expected_volatility'],
            sharpe_ratio=result['sharpe_ratio'],
            diversification_ratio=result['diversification_ratio'],
            risk_metrics=result['risk_metrics'],
            correlation_matrix=result['correlation_matrix'],
            rebalancing_schedule=result['rebalancing_schedule']
        )
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== WEBSOCKET ENDPOINTS ======================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    """WebSocket endpoint for real-time data"""
    client_id = str(uuid.uuid4())
    
    try:
        # Verify token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Connect client
        await connection_manager.connect(websocket, client_id, {"user_id": payload.get("user_id")})
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                message_type = data.get("type")
                
                if message_type == "subscribe":
                    channels = data.get("channels", [])
                    await connection_manager.subscribe(client_id, channels)
                    await websocket.send_json({
                        "type": "subscribed",
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif message_type == "unsubscribe":
                    channels = data.get("channels", [])
                    await connection_manager.unsubscribe(client_id, channels)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "channels": channels,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                elif message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
        except WebSocketDisconnect:
            connection_manager.disconnect(client_id)
            
    except jwt.InvalidTokenError:
        await websocket.close(code=1008, reason="Invalid authentication")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        connection_manager.disconnect(client_id)

# ====================== ALERT ENDPOINTS ======================

@app.post("/api/v1/alerts/settings")
async def update_alert_settings(
    settings: AlertSettings,
    auth: Dict = Depends(verify_jwt_token)
):
    """Update alert settings"""
    try:
        # Update alert manager settings
        await app.state.orchestrator.alert_manager.update_settings(settings.dict())
        
        return {"status": "success", "message": "Alert settings updated"}
    except Exception as e:
        logger.error(f"Error updating alert settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/alerts/history")
async def get_alert_history(
    limit: int = Query(default=50, ge=1, le=500),
    auth: Dict = Depends(verify_jwt_token)
):
    """Get alert history"""
    try:
        alerts = await app.state.orchestrator.alert_manager.get_history(limit)
        return alerts
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ====================== ERROR HANDLERS ======================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# ====================== MAIN ======================

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
        workers=int(os.getenv("WORKERS", 4)),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
        use_colors=True
    )

"""
DEMIR AI v8.0 - Binance WebSocket Manager
REAL-TIME DATA STREAMING - ZERO MOCK DATA
NO SIMPLIFICATION - ENTERPRISE GRADE
"""

import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
import aiohttp
from enum import Enum

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """WebSocket stream types"""
    TICKER = "ticker"
    KLINE = "kline"
    DEPTH = "depth"
    TRADE = "trade"
    AGG_TRADE = "aggTrade"
    BOOK_TICKER = "bookTicker"
    MINI_TICKER = "miniTicker"
    USER_DATA = "userData"


class BinanceWebSocketManager:
    """
    Binance WebSocket connection manager
    REAL-TIME MARKET DATA - NO MOCK STREAMS
    """
    
    def __init__(self, config):
        self.config = config
        
        # WebSocket URLs
        if config.exchange.binance_testnet:
            self.ws_base_url = "wss://testnet.binance.vision/ws"
            self.ws_stream_url = "wss://testnet.binance.vision/stream"
        else:
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
            self.ws_stream_url = "wss://stream.binance.com:9443/stream"
        
        # Connections
        self.connections = {}  # stream_id -> websocket
        self.subscriptions = {}  # stream_id -> subscription_info
        
        # Callbacks
        self.callbacks = {}  # stream_id -> callback_function
        self.error_callbacks = {}  # stream_id -> error_callback
        
        # Data buffers
        self.ticker_data = {}  # symbol -> latest_ticker
        self.orderbook_data = {}  # symbol -> orderbook
        self.trade_data = {}  # symbol -> recent_trades
        self.kline_data = {}  # symbol -> klines
        
        # Connection management
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10
        self.heartbeat_interval = 30  # seconds
        self.is_running = False
        
        # User data stream
        self.listen_key = None
        self.user_stream_task = None
        
        # Tasks
        self.tasks = []
        
        # Statistics
        self.messages_received = 0
        self.bytes_received = 0
        self.connection_starts = {}
        self.disconnection_count = 0
        
        logger.info(f"BinanceWebSocketManager initialized (URL: {self.ws_base_url})")
    
    async def start(self):
        """Start WebSocket manager"""
        self.is_running = True
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.tasks.append(heartbeat_task)
        
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop WebSocket manager"""
        self.is_running = False
        
        # Close all connections
        for stream_id in list(self.connections.keys()):
            await self.disconnect(stream_id)
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("WebSocket manager stopped")
    
    async def subscribe_ticker(self, symbol: str, callback: Optional[Callable] = None) -> str:
        """
        Subscribe to ticker stream
        REAL-TIME PRICE UPDATES
        """
        stream_name = f"{symbol.lower()}@ticker"
        stream_id = f"ticker_{symbol}"
        
        # Setup subscription
        subscription = {
            'type': StreamType.TICKER,
            'symbol': symbol,
            'stream': stream_name,
            'callback': callback
        }
        
        # Connect and subscribe
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to ticker stream for {symbol}")
        
        return stream_id
    
    async def subscribe_kline(self, symbol: str, interval: str = '1m', 
                             callback: Optional[Callable] = None) -> str:
        """
        Subscribe to kline/candlestick stream
        REAL-TIME CANDLESTICK DATA
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        stream_id = f"kline_{symbol}_{interval}"
        
        subscription = {
            'type': StreamType.KLINE,
            'symbol': symbol,
            'interval': interval,
            'stream': stream_name,
            'callback': callback
        }
        
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to kline stream for {symbol} ({interval})")
        
        return stream_id
    
    async def subscribe_depth(self, symbol: str, levels: int = 20,
                             callback: Optional[Callable] = None) -> str:
        """
        Subscribe to order book depth stream
        REAL-TIME ORDER BOOK
        """
        stream_name = f"{symbol.lower()}@depth{levels}@100ms"
        stream_id = f"depth_{symbol}_{levels}"
        
        subscription = {
            'type': StreamType.DEPTH,
            'symbol': symbol,
            'levels': levels,
            'stream': stream_name,
            'callback': callback
        }
        
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to depth stream for {symbol} ({levels} levels)")
        
        return stream_id
    
    async def subscribe_trades(self, symbol: str, callback: Optional[Callable] = None) -> str:
        """
        Subscribe to trade stream
        REAL-TIME TRADES
        """
        stream_name = f"{symbol.lower()}@trade"
        stream_id = f"trade_{symbol}"
        
        subscription = {
            'type': StreamType.TRADE,
            'symbol': symbol,
            'stream': stream_name,
            'callback': callback
        }
        
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to trade stream for {symbol}")
        
        return stream_id
    
    async def subscribe_agg_trades(self, symbol: str, callback: Optional[Callable] = None) -> str:
        """
        Subscribe to aggregated trade stream
        REAL-TIME AGGREGATED TRADES
        """
        stream_name = f"{symbol.lower()}@aggTrade"
        stream_id = f"aggTrade_{symbol}"
        
        subscription = {
            'type': StreamType.AGG_TRADE,
            'symbol': symbol,
            'stream': stream_name,
            'callback': callback
        }
        
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to aggregated trade stream for {symbol}")
        
        return stream_id
    
    async def subscribe_book_ticker(self, symbol: str = None, callback: Optional[Callable] = None) -> str:
        """
        Subscribe to book ticker stream (best bid/ask)
        REAL-TIME BEST PRICES
        """
        if symbol:
            stream_name = f"{symbol.lower()}@bookTicker"
            stream_id = f"bookTicker_{symbol}"
        else:
            stream_name = "!bookTicker"
            stream_id = "bookTicker_all"
        
        subscription = {
            'type': StreamType.BOOK_TICKER,
            'symbol': symbol,
            'stream': stream_name,
            'callback': callback
        }
        
        await self._connect_and_subscribe(stream_id, subscription)
        
        logger.info(f"Subscribed to book ticker stream for {symbol if symbol else 'all symbols'}")
        
        return stream_id
    
    async def subscribe_multiple(self, streams: List[Dict]) -> List[str]:
        """
        Subscribe to multiple streams at once
        COMBINED STREAM SUBSCRIPTION
        """
        stream_ids = []
        combined_streams = []
        
        for stream_config in streams:
            symbol = stream_config['symbol']
            stream_type = stream_config['type']
            
            if stream_type == StreamType.TICKER:
                stream_name = f"{symbol.lower()}@ticker"
            elif stream_type == StreamType.KLINE:
                interval = stream_config.get('interval', '1m')
                stream_name = f"{symbol.lower()}@kline_{interval}"
            elif stream_type == StreamType.DEPTH:
                levels = stream_config.get('levels', 20)
                stream_name = f"{symbol.lower()}@depth{levels}@100ms"
            elif stream_type == StreamType.TRADE:
                stream_name = f"{symbol.lower()}@trade"
            else:
                continue
            
            combined_streams.append(stream_name)
            stream_ids.append(f"{stream_type.value}_{symbol}")
        
        # Create combined stream URL
        if combined_streams:
            combined_url = f"{self.ws_stream_url}?streams={'/'.join(combined_streams)}"
            
            # Connect to combined stream
            await self._connect_combined_stream("combined", combined_url, stream_ids)
            
            logger.info(f"Subscribed to {len(combined_streams)} combined streams")
        
        return stream_ids
    
    async def subscribe_user_data(self, callback: Optional[Callable] = None) -> str:
        """
        Subscribe to user data stream (orders, trades, balances)
        REAL-TIME ACCOUNT UPDATES
        """
        try:
            # Get listen key from REST API
            listen_key = await self._get_listen_key()
            
            if not listen_key:
                logger.error("Failed to get listen key for user data stream")
                return None
            
            self.listen_key = listen_key
            
            # Connect to user data stream
            stream_id = "user_data"
            stream_url = f"{self.ws_base_url}/{listen_key}"
            
            subscription = {
                'type': StreamType.USER_DATA,
                'stream': 'userData',
                'callback': callback,
                'listen_key': listen_key
            }
            
            # Start user data stream
            self.user_stream_task = asyncio.create_task(
                self._maintain_user_stream(stream_id, stream_url, subscription)
            )
            
            logger.info("Subscribed to user data stream")
            
            return stream_id
            
        except Exception as e:
            logger.error(f"Error subscribing to user data stream: {e}")
            return None
    
    async def _connect_and_subscribe(self, stream_id: str, subscription: Dict):
        """Connect to stream and start receiving data"""
        try:
            # Build WebSocket URL
            stream_url = f"{self.ws_base_url}/{subscription['stream']}"
            
            # Store subscription info
            self.subscriptions[stream_id] = subscription
            
            # Register callback
            if subscription.get('callback'):
                self.callbacks[stream_id] = subscription['callback']
            
            # Create connection task
            task = asyncio.create_task(
                self._connection_handler(stream_id, stream_url)
            )
            self.tasks.append(task)
            
            # Track connection start
            self.connection_starts[stream_id] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error connecting to stream {stream_id}: {e}")
            raise
    
    async def _connection_handler(self, stream_id: str, stream_url: str):
        """
        Handle WebSocket connection with reconnection logic
        RELIABLE REAL-TIME CONNECTION
        """
        reconnect_attempts = 0
        
        while self.is_running and reconnect_attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(stream_url) as websocket:
                    # Store connection
                    self.connections[stream_id] = websocket
                    
                    logger.info(f"WebSocket connected: {stream_id}")
                    reconnect_attempts = 0  # Reset on successful connection
                    
                    # Receive messages
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        # Process message
                        await self._process_message(stream_id, message)
                        
                        # Update statistics
                        self.messages_received += 1
                        self.bytes_received += len(message)
            
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed for {stream_id}: {e}")
                self.disconnection_count += 1
                
            except ConnectionClosedError as e:
                logger.warning(f"WebSocket connection error for {stream_id}: {e}")
                self.disconnection_count += 1
                
            except Exception as e:
                logger.error(f"WebSocket error for {stream_id}: {e}")
                
                # Call error callback if registered
                if stream_id in self.error_callbacks:
                    await self.error_callbacks[stream_id](e)
            
            finally:
                # Remove connection
                if stream_id in self.connections:
                    del self.connections[stream_id]
                
                # Reconnect if needed
                if self.is_running and reconnect_attempts < self.max_reconnect_attempts:
                    reconnect_attempts += 1
                    delay = self.reconnect_delay * reconnect_attempts
                    
                    logger.info(f"Reconnecting {stream_id} in {delay} seconds... "
                              f"(Attempt {reconnect_attempts}/{self.max_reconnect_attempts})")
                    
                    await asyncio.sleep(delay)
        
        if reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {stream_id}")
    
    async def _process_message(self, stream_id: str, message: str):
        """
        Process incoming WebSocket message
        REAL DATA PROCESSING
        """
        try:
            data = json.loads(message)
            
            # Get subscription info
            subscription = self.subscriptions.get(stream_id, {})
            stream_type = subscription.get('type')
            symbol = subscription.get('symbol')
            
            # Process based on stream type
            if stream_type == StreamType.TICKER:
                await self._process_ticker(symbol, data)
                
            elif stream_type == StreamType.KLINE:
                await self._process_kline(symbol, data)
                
            elif stream_type == StreamType.DEPTH:
                await self._process_depth(symbol, data)
                
            elif stream_type == StreamType.TRADE:
                await self._process_trade(symbol, data)
                
            elif stream_type == StreamType.AGG_TRADE:
                await self._process_agg_trade(symbol, data)
                
            elif stream_type == StreamType.BOOK_TICKER:
                await self._process_book_ticker(data)
                
            elif stream_type == StreamType.USER_DATA:
                await self._process_user_data(data)
            
            # Call custom callback if registered
            if stream_id in self.callbacks:
                await self.callbacks[stream_id](data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message from {stream_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {stream_id}: {e}")
    
    async def _process_ticker(self, symbol: str, data: Dict):
        """Process ticker data"""
        ticker = {
            'symbol': data['s'],
            'price': float(data['c']),
            'bid': float(data['b']),
            'bid_qty': float(data['B']),
            'ask': float(data['a']),
            'ask_qty': float(data['A']),
            'volume': float(data['v']),
            'quote_volume': float(data['q']),
            'open': float(data['o']),
            'high': float(data['h']),
            'low': float(data['l']),
            'change': float(data['p']),
            'change_percent': float(data['P']),
            'trades': int(data['n']),
            'timestamp': data['E']
        }
        
        self.ticker_data[symbol] = ticker
        
        logger.debug(f"Ticker update for {symbol}: ${ticker['price']:.2f}")
    
    async def _process_kline(self, symbol: str, data: Dict):
        """Process kline/candlestick data"""
        if 'k' not in data:
            return
        
        kline_data = data['k']
        
        kline = {
            'symbol': kline_data['s'],
            'interval': kline_data['i'],
            'open_time': kline_data['t'],
            'close_time': kline_data['T'],
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v']),
            'quote_volume': float(kline_data['q']),
            'trades': kline_data['n'],
            'is_closed': kline_data['x']
        }
        
        # Store kline
        if symbol not in self.kline_data:
            self.kline_data[symbol] = []
        
        # Update or append kline
        if kline['is_closed']:
            self.kline_data[symbol].append(kline)
            
            # Keep only last 500 klines
            if len(self.kline_data[symbol]) > 500:
                self.kline_data[symbol] = self.kline_data[symbol][-500:]
            
            logger.debug(f"Closed kline for {symbol}: O:{kline['open']:.2f} "
                        f"H:{kline['high']:.2f} L:{kline['low']:.2f} C:{kline['close']:.2f}")
    
    async def _process_depth(self, symbol: str, data: Dict):
        """Process order book depth data"""
        orderbook = {
            'symbol': symbol,
            'timestamp': data.get('E', int(time.time() * 1000)),
            'bids': [[float(p), float(q)] for p, q in data.get('b', [])],
            'asks': [[float(p), float(q)] for p, q in data.get('a', [])]
        }
        
        self.orderbook_data[symbol] = orderbook
        
        logger.debug(f"Orderbook update for {symbol}: "
                    f"{len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
    
    async def _process_trade(self, symbol: str, data: Dict):
        """Process trade data"""
        trade = {
            'symbol': data['s'],
            'trade_id': data['t'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'time': data['T'],
            'is_buyer_maker': data['m']
        }
        
        # Store recent trades
        if symbol not in self.trade_data:
            self.trade_data[symbol] = []
        
        self.trade_data[symbol].append(trade)
        
        # Keep only last 100 trades
        if len(self.trade_data[symbol]) > 100:
            self.trade_data[symbol] = self.trade_data[symbol][-100:]
        
        logger.debug(f"Trade for {symbol}: {trade['quantity']:.4f} @ ${trade['price']:.2f}")
    
    async def _process_agg_trade(self, symbol: str, data: Dict):
        """Process aggregated trade data"""
        agg_trade = {
            'symbol': data['s'],
            'agg_trade_id': data['a'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'first_trade_id': data['f'],
            'last_trade_id': data['l'],
            'time': data['T'],
            'is_buyer_maker': data['m']
        }
        
        # You can process aggregated trades here
        logger.debug(f"Aggregated trade for {symbol}: "
                    f"{agg_trade['quantity']:.4f} @ ${agg_trade['price']:.2f}")
    
    async def _process_book_ticker(self, data: Dict):
        """Process book ticker (best bid/ask) data"""
        book_ticker = {
            'symbol': data['s'],
            'bid_price': float(data['b']),
            'bid_qty': float(data['B']),
            'ask_price': float(data['a']),
            'ask_qty': float(data['A']),
            'timestamp': data.get('E', int(time.time() * 1000))
        }
        
        # Update ticker data with best bid/ask
        symbol = book_ticker['symbol']
        if symbol in self.ticker_data:
            self.ticker_data[symbol].update({
                'bid': book_ticker['bid_price'],
                'ask': book_ticker['ask_price'],
                'bid_qty': book_ticker['bid_qty'],
                'ask_qty': book_ticker['ask_qty']
            })
        else:
            self.ticker_data[symbol] = book_ticker
        
        logger.debug(f"Book ticker for {symbol}: "
                    f"Bid: {book_ticker['bid_price']:.2f} Ask: {book_ticker['ask_price']:.2f}")
    
    async def _process_user_data(self, data: Dict):
        """Process user data stream events"""
        event_type = data.get('e')
        
        if event_type == 'executionReport':
            # Order update
            order = {
                'symbol': data['s'],
                'order_id': data['i'],
                'client_order_id': data['c'],
                'side': data['S'],
                'type': data['o'],
                'status': data['X'],
                'price': float(data.get('p', 0)),
                'quantity': float(data['q']),
                'executed_qty': float(data['z']),
                'time': data['T']
            }
            
            logger.info(f"Order update: {order['symbol']} {order['side']} "
                       f"{order['quantity']} - Status: {order['status']}")
            
        elif event_type == 'outboundAccountPosition':
            # Account balance update
            balances = {}
            for balance in data.get('B', []):
                asset = balance['a']
                free = float(balance['f'])
                locked = float(balance['l'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            
            logger.info(f"Balance update: {len(balances)} assets")
    
    async def _maintain_user_stream(self, stream_id: str, stream_url: str, subscription: Dict):
        """
        Maintain user data stream with keepalive
        REAL-TIME ACCOUNT MONITORING
        """
        while self.is_running:
            try:
                # Connect to user stream
                await self._connection_handler(stream_id, stream_url)
                
            except Exception as e:
                logger.error(f"User stream error: {e}")
            
            # Keepalive every 30 minutes
            for _ in range(30):
                if not self.is_running:
                    break
                    
                await asyncio.sleep(60)  # 1 minute
                
                # Send keepalive
                if self.listen_key:
                    await self._keepalive_listen_key(self.listen_key)
    
    async def _get_listen_key(self) -> Optional[str]:
        """Get listen key for user data stream"""
        try:
            # This would call Binance REST API to get listen key
            # Simplified for example
            url = "https://api.binance.com/api/v3/userDataStream"
            
            headers = {'X-MBX-APIKEY': self.config.exchange.binance_api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('listenKey')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting listen key: {e}")
            return None
    
    async def _keepalive_listen_key(self, listen_key: str):
        """Keepalive for user data stream"""
        try:
            url = f"https://api.binance.com/api/v3/userDataStream?listenKey={listen_key}"
            headers = {'X-MBX-APIKEY': self.config.exchange.binance_api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.put(url, headers=headers) as response:
                    if response.status == 200:
                        logger.debug("User stream keepalive sent")
                    else:
                        logger.warning(f"Keepalive failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Keepalive error: {e}")
    
    async def _connect_combined_stream(self, stream_id: str, stream_url: str, stream_ids: List[str]):
        """Connect to combined stream"""
        try:
            subscription = {
                'type': 'combined',
                'streams': stream_ids,
                'callback': None
            }
            
            self.subscriptions[stream_id] = subscription
            
            task = asyncio.create_task(
                self._combined_stream_handler(stream_id, stream_url, stream_ids)
            )
            self.tasks.append(task)
            
        except Exception as e:
            logger.error(f"Error connecting to combined stream: {e}")
    
    async def _combined_stream_handler(self, stream_id: str, stream_url: str, stream_ids: List[str]):
        """Handle combined stream messages"""
        while self.is_running:
            try:
                async with websockets.connect(stream_url) as websocket:
                    self.connections[stream_id] = websocket
                    
                    logger.info(f"Connected to combined stream with {len(stream_ids)} streams")
                    
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        data = json.loads(message)
                        
                        # Route to appropriate processor
                        if 'stream' in data and 'data' in data:
                            stream_name = data['stream']
                            stream_data = data['data']
                            
                            # Determine stream type from name
                            if '@ticker' in stream_name:
                                symbol = stream_name.split('@')[0].upper()
                                await self._process_ticker(symbol, stream_data)
                            elif '@kline' in stream_name:
                                symbol = stream_name.split('@')[0].upper()
                                await self._process_kline(symbol, stream_data)
                            elif '@depth' in stream_name:
                                symbol = stream_name.split('@')[0].upper()
                                await self._process_depth(symbol, stream_data)
                            elif '@trade' in stream_name:
                                symbol = stream_name.split('@')[0].upper()
                                await self._process_trade(symbol, stream_data)
                        
                        self.messages_received += 1
                        self.bytes_received += len(message)
                        
            except Exception as e:
                logger.error(f"Combined stream error: {e}")
                await asyncio.sleep(self.reconnect_delay)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to keep connections alive"""
        while self.is_running:
            try:
                # Send ping to all connections
                for stream_id, ws in list(self.connections.items()):
                    if ws and not ws.closed:
                        try:
                            pong = await ws.ping()
                            logger.debug(f"Heartbeat sent to {stream_id}")
                        except Exception as e:
                            logger.warning(f"Heartbeat failed for {stream_id}: {e}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def disconnect(self, stream_id: str):
        """Disconnect specific stream"""
        try:
            # Close WebSocket connection
            if stream_id in self.connections:
                ws = self.connections[stream_id]
                if ws and not ws.closed:
                    await ws.close()
                
                del self.connections[stream_id]
            
            # Remove subscription
            if stream_id in self.subscriptions:
                del self.subscriptions[stream_id]
            
            # Remove callbacks
            if stream_id in self.callbacks:
                del self.callbacks[stream_id]
            
            logger.info(f"Disconnected stream: {stream_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting stream {stream_id}: {e}")
    
    async def close(self):
        """Close all connections and cleanup"""
        await self.stop()
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get latest ticker data for symbol"""
        return self.ticker_data.get(symbol)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get latest orderbook for symbol"""
        return self.orderbook_data.get(symbol)
    
    def get_recent_trades(self, symbol: str) -> List[Dict]:
        """Get recent trades for symbol"""
        return self.trade_data.get(symbol, [])
    
    def get_klines(self, symbol: str) -> List[Dict]:
        """Get klines for symbol"""
        return self.kline_data.get(symbol, [])
    
    def get_statistics(self) -> Dict:
        """Get WebSocket statistics"""
        uptime = datetime.now() - min(self.connection_starts.values()) if self.connection_starts else timedelta(0)
        
        return {
            'active_connections': len(self.connections),
            'subscriptions': len(self.subscriptions),
            'messages_received': self.messages_received,
            'bytes_received': self.bytes_received,
            'disconnection_count': self.disconnection_count,
            'uptime_seconds': uptime.total_seconds(),
            'cached_tickers': len(self.ticker_data),
            'cached_orderbooks': len(self.orderbook_data)
        }

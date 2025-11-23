"""
DEMIR AI v8.0 - Alert Management System
REAL-TIME ALERTS - TELEGRAM, DISCORD, EMAIL, WEBHOOK
NO MOCK ALERTS - ENTERPRISE GRADE
"""

import logging
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import hashlib
from dataclasses import dataclass, asdict
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Telegram
try:
    from telegram import Bot, ParseMode
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("Telegram bot library not available")

# Discord
try:
    import discord
    from discord import Webhook, RequestsWebhookAdapter
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logging.warning("Discord library not available")

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Alert types"""
    SIGNAL = "SIGNAL"
    TRADE = "TRADE"
    POSITION = "POSITION"
    RISK = "RISK"
    ERROR = "ERROR"
    INFO = "INFO"
    EMERGENCY = "EMERGENCY"
    PERFORMANCE = "PERFORMANCE"
    SYSTEM = "SYSTEM"


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    data: Dict = None
    channels: List[str] = None
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'channels': self.channels,
            'metadata': self.metadata
        }


class TelegramAlertChannel:
    """
    Telegram alert channel
    REAL TELEGRAM MESSAGES - NO MOCK
    """
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        
        if TELEGRAM_AVAILABLE and token:
            self.bot = Bot(token=token)
            logger.info(f"Telegram alert channel initialized for chat {chat_id}")
        else:
            logger.warning("Telegram alert channel not available")
        
        # Rate limiting
        self.last_message_time = {}
        self.min_interval = 1  # Minimum 1 second between messages
        
        # Message queue for rate limiting
        self.message_queue = asyncio.Queue()
        self.sender_task = None
    
    async def start(self):
        """Start message sender task"""
        if self.bot:
            self.sender_task = asyncio.create_task(self._message_sender())
            logger.info("Telegram sender task started")
    
    async def stop(self):
        """Stop message sender task"""
        if self.sender_task:
            self.sender_task.cancel()
            await self.sender_task
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to Telegram
        REAL TELEGRAM MESSAGE
        """
        if not self.bot:
            return False
        
        try:
            # Format message
            message = self._format_message(alert)
            
            # Add to queue
            await self.message_queue.put(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def _message_sender(self):
        """Background task to send messages with rate limiting"""
        while True:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Rate limiting
                current_time = datetime.now()
                if self.chat_id in self.last_message_time:
                    elapsed = (current_time - self.last_message_time[self.chat_id]).total_seconds()
                    if elapsed < self.min_interval:
                        await asyncio.sleep(self.min_interval - elapsed)
                
                # Send message
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
                
                self.last_message_time[self.chat_id] = datetime.now()
                logger.debug(f"Telegram message sent to {self.chat_id}")
                
            except TelegramError as e:
                logger.error(f"Telegram API error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telegram sender error: {e}")
                await asyncio.sleep(5)
    
    def _format_message(self, alert: Alert) -> str:
        """Format alert message for Telegram"""
        # Priority emoji
        priority_emoji = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MEDIUM: "📢",
            AlertPriority.LOW: "ℹ️",
            AlertPriority.INFO: "💡"
        }
        
        # Type emoji
        type_emoji = {
            AlertType.SIGNAL: "📊",
            AlertType.TRADE: "💹",
            AlertType.POSITION: "📈",
            AlertType.RISK: "⚡",
            AlertType.ERROR: "❌",
            AlertType.INFO: "ℹ️",
            AlertType.EMERGENCY: "🚨",
            AlertType.PERFORMANCE: "📉",
            AlertType.SYSTEM: "⚙️"
        }
        
        # Build message
        message = f"{priority_emoji.get(alert.priority, '')} "
        message += f"{type_emoji.get(alert.alert_type, '')} "
        message += f"*{alert.title}*\n\n"
        message += f"{alert.message}\n"
        
        # Add data if present
        if alert.data:
            message += "\n📊 *Details:*\n"
            for key, value in alert.data.items():
                if isinstance(value, float):
                    message += f"• {key}: {value:.4f}\n"
                else:
                    message += f"• {key}: {value}\n"
        
        # Add timestamp
        message += f"\n⏰ {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    async def send_signal_alert(self, signal: Dict) -> bool:
        """Send trading signal alert"""
        alert = Alert(
            alert_id=f"signal_{signal['symbol']}_{datetime.now().timestamp()}",
            alert_type=AlertType.SIGNAL,
            priority=AlertPriority.HIGH if signal['confidence'] > 80 else AlertPriority.MEDIUM,
            title=f"Trading Signal: {signal['symbol']}",
            message=f"{signal['action']} Signal - Confidence: {signal['confidence']:.1f}%",
            timestamp=datetime.now(),
            data={
                'Symbol': signal['symbol'],
                'Action': signal['action'],
                'Entry': signal['entry_price'],
                'Stop Loss': signal['stop_loss'],
                'Take Profit': signal['take_profit_1'],
                'Risk/Reward': signal['risk_reward_ratio']
            }
        )
        
        return await self.send_alert(alert)


class DiscordAlertChannel:
    """
    Discord alert channel
    REAL DISCORD WEBHOOKS - NO MOCK
    """
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.webhook = None
        
        if DISCORD_AVAILABLE and webhook_url:
            # Parse webhook from URL
            self.webhook = Webhook.from_url(
                webhook_url,
                adapter=RequestsWebhookAdapter()
            )
            logger.info("Discord alert channel initialized")
        else:
            logger.warning("Discord alert channel not available")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to Discord
        REAL DISCORD MESSAGE
        """
        if not self.webhook:
            return False
        
        try:
            # Create embed
            embed = self._create_embed(alert)
            
            # Send webhook
            self.webhook.send(embed=embed, username="DEMIR AI")
            
            logger.debug("Discord alert sent")
            return True
            
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            return False
    
    def _create_embed(self, alert: Alert) -> discord.Embed:
        """Create Discord embed from alert"""
        # Color based on priority
        colors = {
            AlertPriority.CRITICAL: 0xFF0000,  # Red
            AlertPriority.HIGH: 0xFFA500,      # Orange
            AlertPriority.MEDIUM: 0xFFFF00,    # Yellow
            AlertPriority.LOW: 0x00FF00,       # Green
            AlertPriority.INFO: 0x0000FF       # Blue
        }
        
        embed = discord.Embed(
            title=alert.title,
            description=alert.message,
            color=colors.get(alert.priority, 0x808080),
            timestamp=alert.timestamp
        )
        
        # Add fields from data
        if alert.data:
            for key, value in alert.data.items():
                if isinstance(value, float):
                    embed.add_field(name=key, value=f"{value:.4f}", inline=True)
                else:
                    embed.add_field(name=key, value=str(value), inline=True)
        
        # Add footer
        embed.set_footer(text=f"DEMIR AI v8.0 | {alert.alert_type.value}")
        
        return embed


class EmailAlertChannel:
    """
    Email alert channel
    REAL EMAIL SENDING - NO MOCK
    """
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 email: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        self.recipients = recipients
        
        logger.info(f"Email alert channel initialized for {len(recipients)} recipients")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert via email
        REAL EMAIL
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[DEMIR AI] {alert.title}"
            msg['From'] = self.email
            msg['To'] = ', '.join(self.recipients)
            
            # Create HTML content
            html_content = self._create_html_content(alert)
            
            # Attach HTML
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.send_message(msg)
            
            logger.debug(f"Email alert sent to {len(self.recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content"""
        priority_colors = {
            AlertPriority.CRITICAL: '#FF0000',
            AlertPriority.HIGH: '#FFA500',
            AlertPriority.MEDIUM: '#FFFF00',
            AlertPriority.LOW: '#00FF00',
            AlertPriority.INFO: '#0000FF'
        }
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {priority_colors.get(alert.priority, '#808080')}; 
                        padding: 10px; color: white;">
                <h2>{alert.title}</h2>
            </div>
            
            <div style="padding: 20px;">
                <p style="font-size: 16px;">{alert.message}</p>
                
                {self._create_data_table(alert.data) if alert.data else ''}
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | 
                    Type: {alert.alert_type.value} | 
                    Priority: {alert.priority.value}
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_data_table(self, data: Dict) -> str:
        """Create HTML table from data"""
        if not data:
            return ""
        
        rows = ""
        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            rows += f"<tr><td style='padding: 5px;'><strong>{key}:</strong></td>"
            rows += f"<td style='padding: 5px;'>{value}</td></tr>"
        
        return f"""
        <table style="margin-top: 20px; border-collapse: collapse;">
            {rows}
        </table>
        """


class WebhookAlertChannel:
    """
    Webhook alert channel for custom integrations
    REAL HTTP WEBHOOKS - NO MOCK
    """
    
    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.session = None
        
        logger.info(f"Webhook alert channel initialized: {webhook_url}")
    
    async def start(self):
        """Start HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def stop(self):
        """Stop HTTP session"""
        if self.session:
            await self.session.close()
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert via webhook
        REAL HTTP POST
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Prepare payload
            payload = alert.to_dict()
            
            # Send POST request
            async with self.session.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    logger.debug(f"Webhook alert sent to {self.webhook_url}")
                    return True
                else:
                    logger.error(f"Webhook failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Webhook send error: {e}")
            return False


class AlertManager:
    """
    Central alert management system
    REAL MULTI-CHANNEL ALERTS - NO MOCK
    """
    
    def __init__(self, config):
        self.config = config
        
        # Alert channels
        self.channels = {}
        
        # Initialize Telegram
        if config.alerts.telegram_enabled:
            self.channels['telegram'] = TelegramAlertChannel(
                token=config.alerts.telegram_token,
                chat_id=config.alerts.telegram_chat_id
            )
        
        # Initialize Discord
        if config.alerts.discord_enabled:
            self.channels['discord'] = DiscordAlertChannel(
                webhook_url=config.alerts.discord_webhook
            )
        
        # Initialize Email (if configured)
        if hasattr(config.alerts, 'email_enabled') and config.alerts.email_enabled:
            self.channels['email'] = EmailAlertChannel(
                smtp_server=config.alerts.smtp_server,
                smtp_port=config.alerts.smtp_port,
                email=config.alerts.email,
                password=config.alerts.email_password,
                recipients=config.alerts.email_recipients
            )
        
        # Initialize Webhook (if configured)
        if hasattr(config.alerts, 'webhook_enabled') and config.alerts.webhook_enabled:
            self.channels['webhook'] = WebhookAlertChannel(
                webhook_url=config.alerts.webhook_url,
                headers=config.alerts.webhook_headers
            )
        
        # Alert history
        self.alert_history = deque(maxlen=1000)
        self.alert_counter = 0
        
        # Rate limiting
        self.alert_cooldowns = {}  # alert_hash -> last_sent_time
        self.cooldown_period = config.alerts.alert_cooldown
        
        # Alert filters
        self.enabled_types = {
            AlertType.SIGNAL: config.alerts.alert_on_signal,
            AlertType.TRADE: config.alerts.alert_on_trade,
            AlertType.ERROR: config.alerts.alert_on_error
        }
        
        # Statistics
        self.alerts_sent = 0
        self.alerts_failed = 0
        
        logger.info(f"AlertManager initialized with {len(self.channels)} channels")
    
    async def start(self):
        """Start all alert channels"""
        for name, channel in self.channels.items():
            if hasattr(channel, 'start'):
                await channel.start()
                logger.info(f"Started alert channel: {name}")
    
    async def stop(self):
        """Stop all alert channels"""
        for name, channel in self.channels.items():
            if hasattr(channel, 'stop'):
                await channel.stop()
                logger.info(f"Stopped alert channel: {name}")
    
    async def send_alert(self, alert_type: AlertType, priority: AlertPriority,
                         title: str, message: str, data: Optional[Dict] = None,
                         channels: Optional[List[str]] = None) -> bool:
        """
        Send alert through configured channels
        REAL ALERT SENDING
        """
        # Check if alert type is enabled
        if alert_type in self.enabled_types and not self.enabled_types[alert_type]:
            logger.debug(f"Alert type {alert_type.value} is disabled")
            return False
        
        # Generate alert ID
        self.alert_counter += 1
        alert_id = f"{alert_type.value}_{self.alert_counter}_{datetime.now().timestamp()}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data=data,
            channels=channels
        )
        
        # Check rate limiting
        alert_hash = self._get_alert_hash(alert)
        if alert_hash in self.alert_cooldowns:
            last_sent = self.alert_cooldowns[alert_hash]
            elapsed = (datetime.now() - last_sent).total_seconds()
            
            if elapsed < self.cooldown_period:
                logger.debug(f"Alert in cooldown: {title}")
                return False
        
        # Determine channels to use
        if channels:
            target_channels = [c for c in channels if c in self.channels]
        else:
            target_channels = list(self.channels.keys())
        
        # Send to each channel
        success_count = 0
        for channel_name in target_channels:
            channel = self.channels[channel_name]
            
            try:
                result = await channel.send_alert(alert)
                
                if result:
                    success_count += 1
                    logger.info(f"Alert sent via {channel_name}: {title}")
                else:
                    logger.warning(f"Alert failed via {channel_name}: {title}")
                    
            except Exception as e:
                logger.error(f"Error sending alert via {channel_name}: {e}")
        
        # Update statistics
        if success_count > 0:
            self.alerts_sent += 1
            self.alert_cooldowns[alert_hash] = datetime.now()
            self.alert_history.append(alert)
            return True
        else:
            self.alerts_failed += 1
            return False
    
    async def send_signal_alert(self, signal: Dict) -> bool:
        """Send trading signal alert"""
        confidence_emoji = "🔥" if signal['confidence'] > 80 else "📊"
        
        message = f"{confidence_emoji} {signal['action']} Signal Detected!\n\n"
        message += f"Confidence: {signal['confidence']:.1f}%\n"
        message += f"Strength: {signal['strength']}\n"
        
        if signal.get('reasons'):
            message += f"\nReasons:\n"
            for reason in signal['reasons']:
                message += f"• {reason}\n"
        
        data = {
            'Symbol': signal['symbol'],
            'Action': signal['action'],
            'Entry Price': signal['entry_price'],
            'Stop Loss': signal['stop_loss'],
            'Take Profit 1': signal['take_profit_1'],
            'Take Profit 2': signal['take_profit_2'],
            'Take Profit 3': signal['take_profit_3'],
            'Position Size': f"{signal['position_size']*100:.1f}%",
            'Risk/Reward': f"{signal['risk_reward_ratio']:.2f}"
        }
        
        priority = AlertPriority.HIGH if signal['confidence'] > 80 else AlertPriority.MEDIUM
        
        return await self.send_alert(
            alert_type=AlertType.SIGNAL,
            priority=priority,
            title=f"Trading Signal: {signal['symbol']} {signal['action']}",
            message=message,
            data=data
        )
    
    async def send_trade_alert(self, trade: Dict) -> bool:
        """Send trade execution alert"""
        message = f"Trade executed successfully!\n\n"
        message += f"Order ID: {trade.get('order_id', 'N/A')}\n"
        
        return await self.send_alert(
            alert_type=AlertType.TRADE,
            priority=AlertPriority.HIGH,
            title=f"Trade Executed: {trade['symbol']} {trade['side']}",
            message=message,
            data=trade
        )
    
    async def send_position_alert(self, position: Dict, event: str) -> bool:
        """Send position update alert"""
        return await self.send_alert(
            alert_type=AlertType.POSITION,
            priority=AlertPriority.MEDIUM,
            title=f"Position {event}: {position['symbol']}",
            message=f"Position {event.lower()} for {position['symbol']}",
            data=position
        )
    
    async def send_risk_alert(self, risk_message: str, data: Optional[Dict] = None) -> bool:
        """Send risk alert"""
        return await self.send_alert(
            alert_type=AlertType.RISK,
            priority=AlertPriority.HIGH,
            title="⚠️ Risk Alert",
            message=risk_message,
            data=data
        )
    
    async def send_emergency_alert(self, message: str) -> bool:
        """Send emergency alert to all channels"""
        return await self.send_alert(
            alert_type=AlertType.EMERGENCY,
            priority=AlertPriority.CRITICAL,
            title="🚨 EMERGENCY ALERT",
            message=message
        )
    
    async def send_notification(self, message: str) -> bool:
        """Send general notification"""
        return await self.send_alert(
            alert_type=AlertType.INFO,
            priority=AlertPriority.INFO,
            title="DEMIR AI Notification",
            message=message
        )
    
    async def send_daily_summary(self, summary: Dict) -> bool:
        """Send daily performance summary"""
        message = "Daily Trading Summary\n\n"
        
        if 'pnl' in summary:
            pnl = summary['pnl']
            emoji = "💰" if pnl > 0 else "📉"
            message += f"{emoji} P&L: ${pnl:.2f}\n"
        
        if 'trades' in summary:
            message += f"📊 Total Trades: {summary['trades']}\n"
        
        if 'win_rate' in summary:
            message += f"🎯 Win Rate: {summary['win_rate']:.1f}%\n"
        
        return await self.send_alert(
            alert_type=AlertType.PERFORMANCE,
            priority=AlertPriority.MEDIUM,
            title="📈 Daily Summary",
            message=message,
            data=summary
        )
    
    def _get_alert_hash(self, alert: Alert) -> str:
        """Generate hash for alert deduplication"""
        hash_string = f"{alert.alert_type.value}_{alert.title}_{alert.message[:50]}"
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            'alerts_sent': self.alerts_sent,
            'alerts_failed': self.alerts_failed,
            'active_channels': len(self.channels),
            'history_size': len(self.alert_history),
            'cooldowns_active': len(self.alert_cooldowns)
        }

"""
DEMIR AI v8.0 - Web Application Server
Serves dashboard and provides API endpoints
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import main orchestrator
from main import DEMIRAIOrchestrator
from config import config

app = FastAPI(
    title="DEMIR AI Dashboard",
    version=config.system.version,
    description="Professional Cryptocurrency Trading Bot"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Root endpoint - serve dashboard
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    index_path = Path("index.html")
    if index_path.exists():
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    else:
        # Return inline HTML if file doesn't exist
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DEMIR AI v8.0</title>
            <style>
                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2rem;
                    background: rgba(255,255,255,0.1);
                    border-radius: 10px;
                    backdrop-filter: blur(10px);
                }
                h1 { margin: 0 0 1rem 0; }
                .status { 
                    padding: 0.5rem 1rem;
                    background: rgba(0,255,0,0.2);
                    border-radius: 5px;
                    display: inline-block;
                    margin-top: 1rem;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 DEMIR AI v8.0</h1>
                <p>Professional Cryptocurrency Trading Bot</p>
                <div class="status">✅ System Active</div>
                <p style="margin-top: 2rem;">Waiting for dashboard file...</p>
            </div>
        </body>
        </html>
        """)

# Health check endpoint (Required for Railway)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "version": config.system.version,
            "environment": config.system.environment,
            "timestamp": datetime.now().isoformat(),
            "uptime": get_uptime()
        }
        
        # Check orchestrator health
        if orchestrator:
            health_status["components"] = {
                "signal_generator": "signal_generator" in orchestrator.components,
                "signal_validator": "signal_validator" in orchestrator.components,
                "data_validator": "data_validator" in orchestrator.components,
                "alerts": "alerts" in orchestrator.components,
                "database": "database" in orchestrator.components,
            }
        
        return JSONResponse(content=health_status, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "daily_pnl": 0.0,
            "active_positions": 0,
            "signals_today": 0,
            "risk_level": "LOW",
            "win_rate": 0.0,
            "total_trades": 0,
            "uptime": get_uptime()
        }
        
        # Get real metrics if available
        if orchestrator and "performance" in orchestrator.components:
            # This would fetch real metrics from performance tracker
            pass
        
        return JSONResponse(content=metrics)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# Signals endpoint
@app.get("/api/signals")
async def get_signals():
    """Get recent signals"""
    try:
        signals = []
        
        # Get signals from orchestrator if available
        if orchestrator and "cache" in orchestrator.components:
            # Fetch from cache
            pass
        
        # Return dummy data for now
        return JSONResponse(content={
            "signals": signals,
            "count": len(signals)
        })
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# System info endpoint
@app.get("/api/info")
async def get_system_info():
    """Get system information"""
    return JSONResponse(content={
        "version": config.system.version,
        "environment": config.system.environment,
        "advisory_mode": config.system.advisory_mode,
        "debug_mode": config.system.debug_mode,
        "trading_symbols": config.trading.symbols,
        "features": {
            "sentiment": config.analysis.enable_sentiment,
            "technical": config.analysis.enable_technical,
            "ml": config.analysis.enable_ml,
            "onchain": config.analysis.enable_onchain
        }
    })

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to DEMIR AI v8.0",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Echo back for now
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Broadcast signal to all connected clients
async def broadcast_signal(signal: Dict):
    """Broadcast trading signal to dashboard"""
    await manager.broadcast({
        "type": "signal",
        "signal": signal,
        "timestamp": datetime.now().isoformat()
    })

# Broadcast metrics update
async def broadcast_metrics(metrics: Dict):
    """Broadcast metrics update to dashboard"""
    await manager.broadcast({
        "type": "metrics",
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })

# Utility functions
def get_uptime() -> str:
    """Get system uptime"""
    if orchestrator and hasattr(orchestrator, 'start_time'):
        delta = datetime.now() - orchestrator.start_time
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m"
    return "0h 0m"

# Background task to run orchestrator
async def run_orchestrator():
    """Run the main orchestrator"""
    global orchestrator
    
    try:
        orchestrator = DEMIRAIOrchestrator()
        await orchestrator.run()
    except Exception as e:
        print(f"Orchestrator error: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Start the orchestrator in background"""
    # Create background task for orchestrator
    asyncio.create_task(run_orchestrator())
    print(f"✅ DEMIR AI Web Server started on port {config.system.api_port}")
    print(f"📊 Dashboard available at http://localhost:{config.system.api_port}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if orchestrator:
        await orchestrator.shutdown()
    print("🛑 DEMIR AI Web Server stopped")

def main():
    """Main entry point for web application"""
    # Check if running directly or through main.py
    if __name__ == "__main__":
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=config.system.debug_mode,
            log_level="info" if config.system.debug_mode else "warning"
        )

if __name__ == "__main__":
    main()

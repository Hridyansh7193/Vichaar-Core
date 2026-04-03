"""
Vichaar-Core API Server

Research-Grade Multi-Agent RL Simulation.
Start: uvicorn api.server:app --host 0.0.0.0 --port 7860
"""
import logging
from fastapi import FastAPI
from api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Vichaar-Core",
    description="Strategic Multi-Agent RL Environment with CEO, Coordinator, and SafeMode",
    version="2.0.0",
)

app.include_router(router)

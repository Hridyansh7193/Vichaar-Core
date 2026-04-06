"""
Vichaar-Core API Server

Research-Grade Multi-Agent RL Simulation.
Start: uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
import logging
import uvicorn
from fastapi import FastAPI
from server.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Vichaar-Core",
    description="Strategic Multi-Agent RL Environment with CEO, Coordinator, and SafeMode",
    version="2.0.0",
)

app.include_router(router)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == '__main__':
    main()

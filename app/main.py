from fastapi import FastAPI

from app.config import settings
from app.routes import health, predict, recommend, models

import uvicorn


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(recommend.router)
app.include_router(models.router)

@app.get("/", tags=["root"])
async def root():
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "documentation": "/docs",
        "endpoints": {
            "predict": "/predict",
            "recommend": "/recommend",
            "models": "/models",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
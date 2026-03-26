import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

def create_app() -> FastAPI:

    app = FastAPI(
        title="Credit Risk Intelligence API",
        version="1.0.0"
    )

    # Logging
    logging.basicConfig(level=logging.INFO)

    # CORS (important for frontend)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()

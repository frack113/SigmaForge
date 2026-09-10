"""SigmaForge - Application entry point."""

import asyncio
import sys

from src.core.schema_validation import validate_schema_version


def create_app():
    """Create and return the FastAPI application instance."""
    from src.api.routes import register_routes

    app = register_routes()
    return app


if __name__ == "__main__":
    import uvicorn

    # Check schema version before starting the server
    validate_schema_version()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "src.main:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        timeout_graceful_shutdown=5,
        log_level="info",
    )

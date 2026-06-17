"""
Vercel Serverless Entry Point
Wraps the FastAPI backend as a single serverless function.
All /api/* requests are routed here by vercel.json.
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app as fastapi_app


class _StripApiPrefix:
    """ASGI middleware that strips /api prefix before reaching FastAPI routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith("/api"):
                scope = dict(scope)
                scope["path"] = path[4:] or "/"
        await self.app(scope, receive, send)


# Export the ASGI app — Vercel's Python runtime detects this automatically
app = _StripApiPrefix(fastapi_app)

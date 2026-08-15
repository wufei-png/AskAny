"""API module for OpenAI-compatible interface."""

from .server import app, create_app

__all__ = ["app", "create_app"]

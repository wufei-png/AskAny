"""API module for OpenAI-compatible interface."""

from .server import app, create_app

__all__ = ["create_app", "app"]

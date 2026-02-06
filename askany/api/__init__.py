"""API module for OpenAI-compatible interface."""

from .server import create_app, app

__all__ = ["create_app", "app"]

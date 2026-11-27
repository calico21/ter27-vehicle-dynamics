"""Core configuration, lifecycle, and application factory exports."""

from api.core.app_factory import create_app
from api.core.config import ApiSettings, settings
from api.core.lifecycle import lifespan

__all__ = ["ApiSettings", "create_app", "lifespan", "settings"]

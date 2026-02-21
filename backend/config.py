"""Application configuration loaded from environment variables via pydantic-settings."""

from typing import Optional

from pydantic import EmailStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings. Values are loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str

    # ── NCBI PubMed ───────────────────────────────────────────────────────────
    ncbi_api_key: Optional[str] = None
    ncbi_email: str  # Required by NCBI policy

    # ── Semantic Scholar ──────────────────────────────────────────────────────
    s2_api_key: Optional[str] = None

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://litlens:litlens@localhost:5432/litlens"
    use_faiss_fallback: bool = False

    # ── Application ───────────────────────────────────────────────────────────
    backend_url: str = "http://localhost:8000"
    environment: str = "development"
    log_level: str = "INFO"
    port: int = 8000

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        """Return True when running in a production environment."""
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        """Return True when running under pytest."""
        return self.environment == "test"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is one of the standard Python logging levels."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is a known value."""
        valid = {"development", "production", "test"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"environment must be one of {valid}")
        return lower


# Module-level singleton — import this everywhere instead of constructing Settings() directly.
settings = Settings()

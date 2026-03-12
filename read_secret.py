"""
Docker secret reader with env var fallback for local development.

In production (inside Crucible), secrets are mounted at /run/secrets/<name>.
For local dev, falls back to environment variables.
"""

import os
from pathlib import Path


def read_secret(name: str, fallback_env: str | None = None) -> str:
    """Read a Docker secret, with optional env var fallback for local dev.

    Args:
        name: Secret name (maps to /run/secrets/<name>)
        fallback_env: Environment variable name to check if secret file not found

    Returns:
        The secret value as a string

    Raises:
        RuntimeError: If secret not found in either location
    """
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        return secret_path.read_text().strip()
    if fallback_env:
        value = os.environ.get(fallback_env, "")
        if value:
            return value
    raise RuntimeError(f"Secret '{name}' not found at {secret_path}")

"""A tiny .env reader (no dependency).

    KEY=value            # comment
    export KEY="value"   (quotes and a leading `export` are accepted)

Variables already set in the real environment win over the file, and an empty value counts as "not set", so an
untouched .env.example-style file changes nothing. The file is model/.env (git-ignored).
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:                                  # inline comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if value:
            values[key] = value
    return values


def load_dotenv(path=None) -> dict[str, str]:
    """Put the variables of the .env file into os.environ (without overriding existing ones).
    Returns the names -> values it actually set. A missing file is fine."""
    if path is None and os.environ.get("URBAN_AIR_NO_DOTENV"):      # the test suite sets this: it must never use real keys
        return {}
    file = Path(path) if path else ENV_FILE
    if not file.is_file():
        return {}
    applied = {}
    for key, value in parse_env(file.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied

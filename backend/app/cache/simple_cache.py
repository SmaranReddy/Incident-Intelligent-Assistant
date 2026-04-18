from cachetools import TTLCache

from app.core.config import settings


class SimpleCache:
    def __init__(self):
        self._cache = TTLCache(
            maxsize=settings.CACHE_MAX_SIZE,
            ttl=settings.CACHE_TTL_SECONDS,
        )

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, value) -> None:
        self._cache[key] = value

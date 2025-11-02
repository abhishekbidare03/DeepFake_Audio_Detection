import time
from threading import Lock


class SimplePredictionCache:
    """Thread-safe in-memory cache for predictions keyed by file hash.
    This is simple and intended for local/testing use only.
    """
    def __init__(self, ttl_seconds=3600):
        self.ttl = ttl_seconds
        self.store = {}  # key -> (timestamp, value)
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            item = self.store.get(key)
            if not item:
                return None
            ts, val = item
            if time.time() - ts > self.ttl:
                del self.store[key]
                return None
            return val

    def set(self, key, value):
        with self.lock:
            self.store[key] = (time.time(), value)

    def clear(self):
        with self.lock:
            self.store.clear()


# Single global cache instance used by the app
prediction_cache = SimplePredictionCache(ttl_seconds=60 * 60)

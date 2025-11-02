import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()

    @property
    def elapsed(self):
        if self.start is None:
            return 0.0
        return (self.end or time.time()) - self.start


@contextmanager
def timeit():
    t = Timer()
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__(None, None, None)


def log_inference(file_name: str, elapsed: float, extra: dict = None):
    # Minimal logging for now; integrate with real monitoring (Prometheus/W&B) later
    extra = extra or {}
    print(f"[monitor] inference file={file_name} time={elapsed:.4f}s {extra}")

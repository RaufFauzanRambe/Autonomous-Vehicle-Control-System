"""
Helper functions for the Autonomous Vehicle Control System.

Provides common utility patterns: retry decorator, timeout, rate limiter,
singleton, observable pattern, deep get/set, and other reusable primitives.

Usage:
    from utils.helper_functions import retry, timeout, rate_limiter, Singleton, Observable

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def connect_to_sensor():
        ...

    @timeout(seconds=5.0)
    def slow_operation():
        ...

    limiter = rate_limiter(calls=10, period=1.0)
    for item in items:
        limiter.wait()
        process(item)

    class ConfigStore(metaclass=Singleton):
        ...
"""

from __future__ import annotations

import functools
import signal
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Set, Tuple, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[F], F]:
    """Decorator that retries a function on specified exceptions.

    Args:
        max_attempts: Maximum number of call attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier applied to delay after each retry.
        max_delay: Upper bound for the delay.
        exceptions: Exception types that trigger a retry.
        on_retry: Callback invoked as ``on_retry(attempt, exception)`` before each retry.

    Returns:
        Decorated function.

    Example::

        @retry(max_attempts=5, delay=0.5, backoff=2.0)
        def fetch_sensor_data():
            response = requests.get(sensor_url)
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        raise
                    if on_retry is not None:
                        on_retry(attempt, exc)
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)

            # Should not reach here, but just in case
            if last_exception is not None:
                raise last_exception
            return None  # type: ignore[return-value]

        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Timeout decorator
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    """Raised when a function exceeds its allowed execution time."""


def timeout(seconds: float) -> Callable[[F], F]:
    """Decorator that raises :class:`TimeoutError` if the function doesn't
    complete within *seconds*.

    Uses ``signal.alarm`` on POSIX systems. On Windows, falls back to a
    threading-based approach.

    Args:
        seconds: Maximum allowed execution time.

    Example::

        @timeout(seconds=5.0)
        def slow_computation():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: Any = None
            exception: Optional[Exception] = None

            def _target() -> None:
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    exception = exc

            if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):
                # POSIX: use signal-based timeout (only works in main thread)
                def _handler(signum: int, frame: Any) -> None:
                    raise TimeoutError(
                        f"Function '{func.__name__}' timed out after {seconds}s"
                    )

                old_handler = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(int(seconds) if seconds == int(seconds) else int(seconds) + 1)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Threading-based timeout (works in any thread)
                thread = threading.Thread(target=_target, daemon=True)
                thread.start()
                thread.join(timeout=seconds)
                if thread.is_alive():
                    raise TimeoutError(
                        f"Function '{func.__name__}' timed out after {seconds}s"
                    )

            if exception is not None:
                raise exception
            return result

        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter.

    Controls the rate at which operations can be performed.

    Args:
        calls: Maximum number of calls allowed in *period* seconds.
        period: Time window in seconds.
    """

    def __init__(self, calls: int = 10, period: float = 1.0) -> None:
        self.calls = calls
        self.period = period
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a call slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Remove timestamps outside the window
                while self._timestamps and self._timestamps[0] <= now - self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.calls:
                    self._timestamps.append(now)
                    return

                # Calculate sleep time
                sleep_time = self._timestamps[0] + self.period - now

            if sleep_time > 0:
                time.sleep(sleep_time)

    def try_acquire(self) -> bool:
        """Non-blocking attempt to acquire a call slot.

        Returns:
            True if a slot was acquired, False otherwise.
        """
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self.period:
                self._timestamps.popleft()
            if len(self._timestamps) < self.calls:
                self._timestamps.append(now)
                return True
            return False

    @property
    def available(self) -> int:
        """Number of available call slots in the current window."""
        with self._lock:
            now = time.monotonic()
            while self._timestamps and self._timestamps[0] <= now - self.period:
                self._timestamps.popleft()
            return self.calls - len(self._timestamps)


def rate_limiter(calls: int = 10, period: float = 1.0) -> RateLimiter:
    """Factory function for creating a :class:`RateLimiter`."""
    return RateLimiter(calls=calls, period=period)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class Singleton(type):
    """Thread-safe metaclass for the Singleton pattern.

    Usage::

        class Database(metaclass=Singleton):
            def __init__(self):
                self.connection = create_connection()
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def _clear_instance(mcs, cls: type) -> None:
        """Remove a singleton instance (for testing)."""
        with mcs._lock:
            mcs._instances.pop(cls, None)


# ---------------------------------------------------------------------------
# Observable pattern
# ---------------------------------------------------------------------------

class Observable:
    """Simple observable / observer pattern implementation.

    Usage::

        class Sensor(Observable):
            def update(self, data):
                self.notify("data_updated", data)

        sensor = Sensor()
        sensor.subscribe("data_updated", on_new_data)
        sensor.update(latest_reading)
    """

    def __init__(self) -> None:
        self._observers: Dict[str, List[Callable[..., None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, callback: Callable[..., None]) -> None:
        """Register *callback* for *event*."""
        with self._lock:
            self._observers.setdefault(event, []).append(callback)

    def unsubscribe(self, event: str, callback: Callable[..., None]) -> None:
        """Remove *callback* from *event*."""
        with self._lock:
            if event in self._observers:
                self._observers[event] = [
                    cb for cb in self._observers[event] if cb != callback
                ]

    def notify(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Notify all subscribers of *event*."""
        with self._lock:
            callbacks = list(self._observers.get(event, []))
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Observer errors should not crash the subject

    def clear_subscribers(self, event: Optional[str] = None) -> None:
        """Remove all subscribers for *event*, or all events if None."""
        with self._lock:
            if event is None:
                self._observers.clear()
            else:
                self._observers.pop(event, None)


# ---------------------------------------------------------------------------
# Deep dictionary utilities
# ---------------------------------------------------------------------------

def deep_get(data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Retrieve a nested value using dot-separated key notation.

    Args:
        data: Nested dictionary.
        dotted_key: e.g. ``"perception.model.path"``
        default: Value returned if key doesn't exist.

    Returns:
        The found value or *default*.
    """
    keys = dotted_key.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def deep_set(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested value using dot-separated key notation.

    Creates intermediate dictionaries as needed.
    """
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update *base* dict with values from *override*.

    Returns a new dict; does not mutate *base*.
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Memoization with TTL
# ---------------------------------------------------------------------------

class MemoizeTTL:
    """Memoization decorator with a time-to-live (TTL) for cache entries.

    Args:
        ttl: Cache entry lifetime in seconds.
        max_size: Maximum number of entries to cache.

    Example::

        @MemoizeTTL(ttl=60.0, max_size=128)
        def fetch_weather(location):
            return api.get_weather(location)
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 256) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()

            with self._lock:
                if key in self._cache:
                    ts, value = self._cache[key]
                    if now - ts < self.ttl:
                        return value
                    # Expired – remove
                    del self._cache[key]

            # Compute outside lock to avoid contention
            result = func(*args, **kwargs)

            with self._lock:
                self._cache[key] = (now, result)
                # Evict oldest entries if over max_size
                if len(self._cache) > self.max_size:
                    oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                    del self._cache[oldest_key]

            return result

        return wrapper  # type: ignore[return-value]

    def invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Remove a specific entry from cache."""
        key = (args, tuple(sorted(kwargs.items())))
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------

def debounce(wait: float) -> Callable[[F], F]:
    """Decorator that debounces a function – only calls it after *wait*
    seconds of inactivity.

    Useful for event handlers that fire too frequently (e.g., sensor
    callbacks, UI inputs).
    """
    def decorator(func: F) -> F:
        timer: Optional[threading.Timer] = None
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            nonlocal timer

            def _call():
                func(*args, **kwargs)

            with lock:
                if timer is not None:
                    timer.cancel()
                timer = threading.Timer(wait, _call)
                timer.daemon = True
                timer.start()

        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

def throttle(rate: float) -> Callable[[F], F]:
    """Decorator that throttles a function to at most one call per *rate*
    seconds.

    Unlike debounce, throttle guarantees the function is called at a
    regular cadence even under continuous input.
    """
    def decorator(func: F) -> F:
        last_called: List[float] = [0.0]
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.monotonic()
            with lock:
                if now - last_called[0] >= rate:
                    last_called[0] = now
                    return func(*args, **kwargs)
            return None

        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Enum helpers
# ---------------------------------------------------------------------------

def enum_to_dict(enum_class: type) -> Dict[str, Any]:
    """Convert an Enum class to a {name: value} dictionary."""
    return {item.name: item.value for item in enum_class}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunked(iterable: Sequence[T], size: int) -> List[Sequence[T]]:
    """Split *iterable* into chunks of *size*."""
    return [iterable[i:i + size] for i in range(0, len(iterable), size)]


# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------

def safe_execute(
    func: Callable[..., T],
    *args: Any,
    default: Any = None,
    log_error: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute *func* and return *default* on any exception.

    Args:
        func: Function to execute.
        *args: Positional arguments for *func*.
        default: Value to return on failure.
        log_error: If True, log the exception.
        **kwargs: Keyword arguments for *func*.

    Returns:
        Function result or *default*.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if log_error:
            import logging
            logging.getLogger("helper_functions").warning(
                "safe_execute caught exception in %s: %s", func.__name__, exc
            )
        return default

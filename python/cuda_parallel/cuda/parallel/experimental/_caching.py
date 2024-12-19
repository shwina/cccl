import functools


def cache_with_key(key):
    """
    Cache the result of `func`, using the function `key` to compute
    the key for cache lookup. `key` receives all arguments passed to
    `func`.
    """

    def deco(func):
        cache = {}

        @functools.wraps(func)
        def inner(*args, **kwargs):
            cache_key = key(*args, **kwargs)
            if cache_key not in cache:
                result = func(*args, **kwargs)
                cache[cache_key] = result
            # `cache_key` *must* be in `cache`, use `.get()`
            # as it is faster:
            return cache.get(cache_key)

        return inner

    return deco

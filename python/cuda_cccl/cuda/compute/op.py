from ._bindings import OpKind


class StatefulOp:
    """
    Wraps a callable with device state for use in CCCL algorithms.

    A stateful operator maintains device memory that can be read and modified
    during algorithm execution. This is useful for operations like counting
    rejected items during filtering, accumulating statistics, etc.

    Args:
        state: Device array containing the state. Must be contiguous device memory.
        func: Callable that takes state_ptr as first argument, followed by algorithm-specific
              arguments. The state_ptr is a CPointer to the state in device global memory.
              For example, a filter predicate would be: func(state_ptr, x) -> bool

    Note:
        The state is passed as a CPointer (raw pointer) to the device function. You can:

        - Access individual elements: ``state_ptr[0]``, ``state_ptr[1]``, etc.
        - Create an array view for atomics: ``state_array = carray(state_ptr, (size,))``
          then use ``cuda.atomic.add(state_array, 0, 1)``

    Example (reading state):
        >>> import cupy as cp
        >>> import numpy as np
        >>> from numba import cuda
        >>> state = cp.array([50], dtype=np.int32)
        >>>
        >>> def threshold_filter(state_ptr, x):
        ...     threshold = state_ptr[0]  # Direct pointer access
        ...     return x > threshold
        >>>
        >>> op = StatefulOp(state, threshold_filter)
        >>> # Use with filter, reduce, etc.

    Example (modifying state):
        >>> state = cp.zeros(2, dtype=np.int32)
        >>>
        >>> def track_values(state_ptr, x):
        ...     # Direct pointer access for writes (atomic-safe for single thread)
        ...     if x > 10:
        ...         state_ptr[0] += 1  # count of values > 10
        ...         return True
        ...     else:
        ...         state_ptr[1] += 1  # count of values <= 10
        ...         return False
        >>>
        >>> op = StatefulOp(state, track_values)
        >>> # After execution, check state[0] and state[1] for counts

    Note:
        For concurrent writes from multiple threads, atomic operations may be needed.
        However, integrating numba.cuda.atomic with carray in device functions
        compiled through intrinsics has limitations. For simple counters or statistics,
        consider using dedicated output arrays instead of stateful operators.
    """

    def __init__(self, state, func):
        """
        Initialize a stateful operator.

        Args:
            state: Device array containing the state
            func: Callable with state as first parameter
        """
        self._state_array = state  # Keep reference to keep memory alive
        self._func = func

    @property
    def state(self):
        """Access the device state array."""
        return self._state_array

    @property
    def func(self):
        """Access the wrapped callable."""
        return self._func


__all__ = ["OpKind", "StatefulOp"]

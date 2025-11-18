from functools import wraps, lru_cache

import numpy as np
from xxhash import xxh3_128_hexdigest


class NumpyArrayListHasher:
    def __init__(self, x) -> None:
        self.values = x
        self.h = self._create_hashable(x)
        self._hash_val = hash(self.h)

    def _create_hashable(self, list_of_np_arrays):
        hashable = sorted(xxh3_128_hexdigest(arr) for arr in list_of_np_arrays)
        ret = tuple(hashable)
        return ret

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h

    def clear_inputs(self):
        self.values = None


def list_of_np_cache(*args, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(list_of_np_array, *args, **kwargs):
            wrapped_obj = NumpyArrayListHasher(list_of_np_array)
            return cached_wrapper(wrapped_obj, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_set, *args, **kwargs):
            array = hashable_set.values
            hashable_set.clear_inputs()
            return function(array, *args, **kwargs)

        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


class NumpyTwoArraysHasher:
    def __init__(self, arr_1, arr_2) -> None:
        self.input_1 = arr_1
        self.input_2 = arr_2
        self.h = frozenset([xxh3_128_hexdigest(arr_1), xxh3_128_hexdigest(arr_2)])
        self._hash_val = hash(self.h)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h

    def clear_inputs(self):
        self.input_1 = None
        self.input_2 = None


def two_np_arr_cache(*args, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(arr_1, arr_2, *args, **kwargs):
            wrapped_obj = NumpyTwoArraysHasher(arr_1, arr_2)
            return cached_wrapper(wrapped_obj, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_obj, *args, **kwargs):
            arr_1 = hashable_obj.input_1
            arr_2 = hashable_obj.input_2
            hashable_obj.clear_inputs()
            return function(arr_1, arr_2, *args, **kwargs)

        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


class NumpyArrayHasher:
    def __init__(self, arr_1) -> None:
        self.input = arr_1
        self.h = xxh3_128_hexdigest(arr_1)

    def __hash__(self) -> int:
        return hash(self.h)

    def __eq__(self, __value: object) -> bool:
        return __value.h == self.h

    def clear_inputs(self):
        self.input = None


def fxn_with_np_array_cache(*args, **kwargs):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            new_args = []

            for arg in args:
                if isinstance(arg, np.ndarray):
                    new_args.append(NumpyArrayHasher(arg))
                else:
                    new_args.append(arg)

            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    kwargs[k] = NumpyArrayHasher(v)

            return cached_wrapper(*new_args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(*args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, NumpyArrayHasher):
                    new_args.append(arg.input)
                    arg.clear_inputs()
                else:
                    new_args.append(arg)

            for k, v in kwargs.items():
                if isinstance(v, NumpyArrayHasher):
                    kwargs[k] = v.input
                    v.clear_inputs()

            return function(*new_args, **kwargs)

        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator

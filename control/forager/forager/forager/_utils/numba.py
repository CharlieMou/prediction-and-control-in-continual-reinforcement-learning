import functools as ft
import typing as typ

try:
    import numba as nb
except ModuleNotFoundError:
    class _FallbackDict(dict):
        @classmethod
        def empty(cls, **_kwargs):
            return cls()

    class _FallbackTyped:
        Dict = _FallbackDict
        List = list

    class _FallbackType:
        def __getitem__(self, _item):
            return self

    class _FallbackTypes:
        int64 = _FallbackType()
        uint8 = _FallbackType()

    class _FallbackNumba:
        """Small compatibility layer for running Forager without Numba."""

        typed = _FallbackTyped()
        types = _FallbackTypes()

        @staticmethod
        def typeof(value):
            return type(value)

        @staticmethod
        def njit(*args, **_kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]

            def decorator(function):
                return function

            return decorator

    nb = _FallbackNumba()

njit = ft.partial(nb.njit, cache=True, fastmath=True, nogil=True)

# TODO: pylance and mypy aren't aware of these types
# hiding them here to make tooling behave better.
# Could consider my own type-stubs later
Dict: typ.Any = nb.typed.Dict
List: typ.Any = nb.typed.List
typeof: typ.Any = nb.typeof
types: typ.Any = nb.types


@njit(inline='always')
def ravel(coords: typ.Tuple[int, int], size: typ.Tuple[int, int]) -> int:
    return coords[0] + coords[1] * size[0]

@njit(inline='always')
def unravel(idx: int, size: typ.Tuple[int, int]) -> typ.Tuple[int, int]:
    x = int(idx % size[0])
    y = int(idx // size[0])

    return (x, y)

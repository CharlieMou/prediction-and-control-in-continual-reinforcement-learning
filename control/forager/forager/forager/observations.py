import numpy as np
import forager._utils.numba as nbu

from typing import Dict
from forager.interface import Coords, Size


@nbu.njit
def get_color_vision(
    state: Coords,
    size: Size,
    ap_size: Size,
    idx_to_name: Dict[int, str],
    name_to_color: Dict[str, np.ndarray]
) -> np.ndarray:
    out = np.zeros((ap_size[0], ap_size[1], 3), dtype=np.uint8)

    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    xs = range(state[0] - ax, state[0] + ax + 1)
    ys = range(state[1] - ay, state[1] + ay + 1)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            x = x % size[0]
            y = y % size[1]

            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in idx_to_name:
                name = idx_to_name[idx]
                color = name_to_color[name]
                jr = ap_size[1] - j - 1
                out[jr, i] = color

    return out


@nbu.njit
def render_aperture(
    state: Coords,
    size: Size,
    ap_size: Size,
    idx_to_name: Dict[int, str],
    name_to_color: Dict[str, np.ndarray],
    agent_color: np.ndarray
) -> np.ndarray:
    out = np.ones((ap_size[0], ap_size[1], 3), dtype=np.uint8) * 255

    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    xs = range(state[0] - ax, state[0] + ax + 1)
    ys = range(state[1] - ay, state[1] + ay + 1)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            x = x % size[0]
            y = y % size[1]

            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in idx_to_name:
                name = idx_to_name[idx]
                color = name_to_color[name]
                jr = ap_size[1] - j - 1
                out[jr, i] = color

    out[ap_size[1] // 2, ap_size[0] // 2] = agent_color

    return out


@nbu.njit
def get_object_vision(
    state: Coords,
    size: Size,
    ap_size: Size,
    objs: Dict[Coords, str],
    names: Dict[str, int],
) -> np.ndarray:
    dims = len(names)
    out = np.zeros((ap_size[0], ap_size[1], dims), dtype=np.bool_)

    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    xs = range(state[0] - ax, state[0] + ax + 1)
    ys = range(state[1] - ay, state[1] + ay + 1)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            x = x % size[0]
            y = y % size[1]

            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in objs:
                obj = objs[idx]
                d = names[obj]
                jr = ap_size[1] - j - 1
                out[jr, i, d] = 1

    return out

@nbu.njit
def get_world_vision(
    state: Coords,
    size: Size,
    objs: Dict[Coords, str],
    names: Dict[str, int],
) -> np.ndarray:
    dims = len(names)
    out = np.zeros((size[1], size[0], dims), dtype=np.bool_)

    for i, x in enumerate(range(size[0])):
        for j, y in enumerate(range(size[1])):
            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in objs:
                obj = objs[idx]
                d = names[obj]
                out[j, i, d] = 1

    # agent_dim
    b_dim = names['agent']
    out[state[1], state[0], b_dim] = 1
    return out

@nbu.njit
def render_world(
    state: Coords,
    size: Size,
    ap_size: Size,
    idx_to_name: Dict[int, str],
    name_to_color: Dict[str, np.ndarray],
    agent_color: np.ndarray
) -> np.ndarray:
    out = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    for x in range(size[0]):
        for y in range(size[1]):
            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in idx_to_name:
                name = idx_to_name[idx]
                color = name_to_color[name]
                out[y, x] = color

    # draw agent
    out[state[1], state[0]] = agent_color
    alpha = 0.2

    # draw agent aperture
    if ap_size is not None:
        alpha = 0.2
        ax = int(ap_size[0] // 2)
        ay = int(ap_size[1] // 2)

        xs = range(state[0] - ax, state[0] + ax + 1)
        ys = range(state[1] - ay, state[1] + ay + 1)

        for x in xs:
            for y in ys:
                x = x % size[0]
                y = y % size[1]
                c = (x, y)
                idx = nbu.ravel(c, size)
                out[y, x] = (1 - alpha) * out[y, x] + alpha * agent_color

    return out

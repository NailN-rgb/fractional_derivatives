import numpy as np
from .primitives import Field, Field1D, Field2D

from ..forward.solution import Solution
from ..protocols import Grid
from ..types import Array
from ..verification.reference import ReferenceSolution

def _shaped(grid: Grid, flat: Array) -> Array:
    return np.asarray(flat).reshape(grid.shape)

def _wrap(
    g: Grid,
    values: Array,
    *,
    label=None,
    signed=False,
    vlabel='u'
) -> Field:
    if values.ndim == 1:
        return Field1D(
            x=g.nodes[0], values=values, xlabel=g.axes[0], vlabel=vlabel, label=label
        )

    if values.ndim == 2:
        return Field2D(
            x=g.nodes[0], y=g.nodes[1], values=values.T, xlabel=g.axes[0], ylabel=g.axes[1],
            vlabel=vlabel, label=label
        )

    raise NotImplementedError("График для таких данных построить невозможно")

def at(
    sol: Solution, 
    t: float, 
    label=None, 
    signed: bool=False, 
    vlabel: str='u'
) -> Field:
    lbl = label if label is not None else f'$t={t:g}$'
    g = sol.space
    return _wrap(g, _shaped(g, sol.at_time(t)), label=lbl, signed=signed, vlabel=vlabel)


def exact(
    ref: ReferenceSolution,
    grid: Grid,
    t: float,
    label=None,
    signed: bool=False,
    vlabel: str='u'
) -> Field:
    lbl = label if label is not None else f'exact, $t={t:g}$'
    return _wrap(grid, _shaped(grid, ref.exact(grid.coords, t, grid)),
                 label=lbl, signed=signed, vlabel=vlabel)


def error(
    sol: Solution,
    ref: ReferenceSolution,
    t: float,
    label=None,
    vlabel: str=r'$u_h - u$'
) -> Field:
    """Поточечная невязка — тот же Field, рисуется тем же plot()."""
    g = sol.space
    diff = sol.at_time(t) - ref.exact(g.coords, t, g)
    lbl = label if label is not None else f'error, $t={t:g}$'
    return _wrap(g, _shaped(g, diff), label=lbl, signed=True, vlabel=vlabel)

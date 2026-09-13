import numpy as np
from .primitives import Field1D, Field2D

from ..forward.solution import Solution
from ..types import Array

def _shaped(sol: Solution, flat: Array):
    return np.asarray(flat).reshape(sol.space.shape)

def _wrap(sol: Solution, values: Array, *, label=None, signed=False, vlabel='u'):
    g = sol.space

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

def at(sol: Solution, t: float, label=None, signed: bool=False, vlabel: str='u'):
    lbl = label if label is not None else f'$t={t:g}$'

    return _wrap(sol, _shaped(sol, sol.at_time(t)), label=lbl, signed=signed, vlabel=vlabel)



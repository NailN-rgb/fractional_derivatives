import numpy as np
import matplotlib.pyplot as plt

from .primitives import Field1D, Field2D

def _axes(axes):
    return plt.gca() if axes is None else axes

def plot_field1d(f1d: Field1D, axes, **kw):
    axes = _axes(axes)
    kw.setdefault("label", f1d.label)

    (line,) = axes.plot(f1d.x, f1d.values, **kw)

    axes.set_xlabel(f1d.xlabel)
    axes.set_ylabel(f1d.vlabel)
    return line 

def plot_field2d(obj, axes, **kw):
    raise NotImplementedError("2d field visualization is not implemented yet")

# Main plot func
def plot(obj, axes=None, **kw):
    axes = _axes(axes)

    if isinstance(obj, Field1D):
        return plot_field1d(obj, axes, **kw)
    if isinstance(obj, Field2D):
        return plot_field2d(obj, axes, **kw)

    raise NotImplementedError("Unknown graph type")


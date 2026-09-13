from dataclasses import dataclass

from ..types import Array

@dataclass(frozen=True, slots=True)
class Field1D:
    x: Array
    values: Array
    xlabel: str = 'x'
    vlabel: str = 'u'
    label: str | None = None

@dataclass(frozen=True, slots=True)
class Field2D:
    x: Array
    y: Array
    values: Array
    xlabel: str = 'x'
    ylabel: str = 'y'
    vlabel: str = 'u'
    label: str | None = None

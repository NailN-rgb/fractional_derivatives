from dataclasses import dataclass

from ..types import Array

@dataclass(frozen=True, slots=True, kw_only=True)
class Field:
    values: Array
    vlabel: str = 'u'
    label: str | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class Field1D(Field):
    x: Array
    values: Array
    xlabel: str = 'x'
    vlabel: str = 'u'
    label: str | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class Field2D(Field):
    x: Array
    y: Array
    values: Array
    xlabel: str = 'x'
    ylabel: str = 'y'
    vlabel: str = 'u'
    label: str | None = None

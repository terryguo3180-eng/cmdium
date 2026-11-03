from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, cast


class DataEnum(Enum):
  def __repr__(self):
    return f"{type(self).__name__}.{self.name}"


def _is_node(v: object) -> bool:
  return is_dataclass(v) and not isinstance(v, type)


def _is_leaf(v: object) -> bool:
  return not _is_node(v) and not isinstance(v, list)


def _shown(node: Any) -> list[tuple[str, object]]:
  return [(f.name, getattr(node, f.name)) for f in fields(node) if f.repr]


def _leaf(v: object) -> str:
  return v.name if isinstance(v, Enum) else repr(v)


def _flat(v: object) -> str:
  if _is_node(v):
    name, items = type(v).__name__, _shown(v)
    if len(items) == 1 and _is_leaf(items[0][1]):
      return f"({name} {_leaf(items[0][1])})"
    body = " ".join(f"{k}={_flat(x)}" for k, x in items)
    return f"({name} {body})" if body else f"({name})"
  if isinstance(v, list):
    return "[" + " ".join(_flat(x) for x in cast("list[object]", v)) + "]"
  return _leaf(v)


def _fmt(v: object, col: int, width: int) -> str:
  flat = _flat(v)
  if col + len(flat) <= width or _is_leaf(v):
    return flat

  if isinstance(v, list):
    pad = "\n" + " " * (col + 1)
    items = cast("list[object]", v)
    return "[" + pad.join(_fmt(x, col + 1, width) for x in items) + "]"

  pad = " " * (col + 2)
  lines = [f"({type(v).__name__}"]
  for k, x in _shown(v):
    lines.append(pad + f"{k}=" + _fmt(x, col + 2 + len(k) + 1, width))
  return "\n".join(lines) + ")"


def dump(node: object, width: int = 80) -> str:
  return _fmt(node, 0, width)


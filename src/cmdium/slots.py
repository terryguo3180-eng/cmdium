from __future__ import annotations

from dataclasses import dataclass
from enum import auto

from cmdium import lowering
from cmdium.utils import DataEnum


class Store(DataEnum):
  """The two places a value can live in a data pack."""

  SCORE = auto()  # a scoreboard score: <holder> <objective>
  NBT = auto()    # an NBT tag: storage <namespace>:<id> <path>


def store_of(type: lowering.ScalarType) -> Store:
  match type:
    case (
      lowering.ScalarType.STR |
      lowering.ScalarType.LABEL |
      lowering.ScalarType.FUNC |
      lowering.ScalarType.PTR
    ):
      return Store.NBT

    case lowering.ScalarType.INT:
      return Store.SCORE


@dataclass(frozen=True)
class Lit:
  """A compile-time scalar constant."""

  type: lowering.ScalarType
  value: int | str

  @property
  def store(self) -> Store:
    return store_of(self.type)

  def as_int(self) -> int:
    assert isinstance(self.value, int)
    return self.value


@dataclass(frozen=True)
class Slot:
  """One globally addressable storage cell.

  `id` is only meaningful within a program; `store` selects how it is
  addressed, and the remaining fields are declarative metadata for the ABI
  (an optional preferred name, whether the name is public, and a known
  constant).  A slot without a hint is named after its id.
  """

  id: int
  store: Store
  hint: str | None
  export_name: str | None = None
  const: Lit | None = None


# a value in codegen: either a storage cell or an inline constant
Value = Slot | Lit


class SlotTable:
  """Assigns one slot per scalar value and interns constants."""

  def __init__(self, exports: dict[lowering.Var, str]):
    self.slots: list[Slot] = []
    self.exports = exports
    self._vars: dict[lowering.Var, Slot] = {}
    self._consts: dict[Lit, Slot] = {}

  def alloc(
    self,
    store: Store,
    hint: str | None = None,
    export_name: str | None = None,
    const: Lit | None = None,
  ) -> Slot:
    slot = Slot(len(self.slots), store, hint, export_name, const)
    self.slots.append(slot)
    return slot

  def of_var(self, var: lowering.Var) -> Slot:
    slot = self._vars.get(var)
    if slot is None:
      export_name = self.exports.get(var)
      slot = self.alloc(store_of(var.type), var.hint, export_name)
      self._vars[var] = slot
    return slot

  def const(self, lit: Lit) -> Slot:
    """Return the slot holding `lit`, creating it on first use."""
    slot = self._consts.get(lit)
    if slot is None:
      slot = self.alloc(lit.store, str(lit.value), const=lit)
      self._consts[lit] = slot
    return slot

  def consts(self) -> list[Slot]:
    return [slot for slot in self.slots if slot.const is not None]

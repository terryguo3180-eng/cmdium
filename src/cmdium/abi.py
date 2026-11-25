from __future__ import annotations

import json
import re

from enum import auto

from cmdium import lowering
from cmdium.slots import Lit, Slot, SlotTable, Store
from cmdium.utils import DataEnum


class CmpOp(DataEnum):
  EQ = auto()
  NE = auto()
  LT = auto()
  LE = auto()
  GT = auto()
  GE = auto()


CMP_OPS = {
  lowering.BinaryOp.CEQ: CmpOp.EQ,
  lowering.BinaryOp.CNE: CmpOp.NE,
  lowering.BinaryOp.CLT: CmpOp.LT,
  lowering.BinaryOp.CLE: CmpOp.LE,
  lowering.BinaryOp.CGT: CmpOp.GT,
  lowering.BinaryOp.CGE: CmpOp.GE,
}

ARITH_OPS = {
  lowering.BinaryOp.ADD: "+=",
  lowering.BinaryOp.SUB: "-=",
  lowering.BinaryOp.MUL: "*=",
  lowering.BinaryOp.DIV: "/=",
  lowering.BinaryOp.MOD: "%=",
  lowering.BinaryOp.OR: ">",   # choosing maximum
  lowering.BinaryOp.AND: "<",  # choosing minimum
}

CMP_SYMBOLS = {
  CmpOp.LT: "<",
  CmpOp.LE: "<=",
  CmpOp.GT: ">",
  CmpOp.GE: ">=",
}


class ABI:
  """Java Edition syntax and naming for slots.

  Everything that has to know how a value is spelled in a data pack lives
  here; the rest of codegen only deals in slots.
  """

  def __init__(self, namespace: str, objective: str, slots: SlotTable):
    self.namespace = namespace
    self.objective = objective
    self.slots = slots

    self._names: dict[Slot, str] = {}
    self._bases: dict[Slot, str] = {}
    self._counts: dict[str, int] = {}
    # NBT equality of two slots needs somewhere to stage a copy
    self._scratch = slots.alloc(Store.NBT)

  # -- naming --------------------------------------------------------------
  def _unique(self, hint: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.+\-]", "_", hint)
    n = self._counts.get(base, 0)
    self._counts[base] = n + 1
    return base if n == 0 else f"{base}{n}"

  # a macro variable has to be a plain word; anything else (aggregate members
  # keep dots, for instance) is not usable as `$(name)`
  MACRO_NAME = re.compile(r"[A-Za-z0-9_]+")

  @classmethod
  def is_macro_name(cls, name: str) -> bool:
    return cls.MACRO_NAME.fullmatch(name) is not None

  def name(self, slot: Slot) -> str:
    name = self._names.get(slot)
    if name is not None:
      return name

    if slot.export_name is not None:
      hint = slot.export_name
    elif slot.hint is not None:
      hint = slot.hint
    else:
      # unnamed internal slots are named after their id
      hint = f"v{slot.id}"

    base = self._unique(hint)
    self._bases[slot] = base

    name = base
    if slot.store is Store.SCORE:
      name = ("$" if slot.export_name is not None else "#") + base

    self._names[slot] = name
    return name

  def base(self, slot: Slot) -> str:
    """The name of a cell without the scoreboard holder prefix.

    A pointer stores this, and `#$(p)` turns it back into a holder.
    """
    self.name(slot)
    return self._bases[slot]

  def container(self) -> str:
    return f"storage {self.namespace}:main"

  def data_path(self, slot: Slot) -> str:
    name = self.name(slot)
    return name if name.isidentifier() else json.dumps(name)

  def score_ref(self, slot: Slot) -> str:
    return f"{self.name(slot)} {self.objective}"

  def data_ref(self, slot: Slot) -> str:
    return f"{self.container()} {self.data_path(slot)}"

  # -- literals ------------------------------------------------------------
  def lit(self, lit: Lit) -> str:
    match lit.type:
      case lowering.ScalarType.STR | lowering.ScalarType.PTR:
        return json.dumps(lit.value, ensure_ascii=False)

      case lowering.ScalarType.FUNC:
        assert isinstance(lit.value, str)
        return json.dumps(self.func_name(lit.value), ensure_ascii=False)

      case lowering.ScalarType.INT:
        return str(lit.value)

      case _:
        raise NotImplementedError(f"literal of type {lit.type!r}")

  # -- scoreboard ----------------------------------------------------------
  def add_objective(self) -> str:
    return f"scoreboard objectives add {self.objective} dummy"

  def set_score(self, dst: Slot, val: int) -> str:
    return f"scoreboard players set {self.score_ref(dst)} {val}"

  def copy_score(self, dst: Slot, src: Slot) -> str:
    return (
      f"scoreboard players operation {self.score_ref(dst)} = "
      f"{self.score_ref(src)}"
    )

  def add_score(self, dst: Slot, val: int) -> str:
    if val < 0:
      return f"scoreboard players remove {self.score_ref(dst)} {-val}"
    return f"scoreboard players add {self.score_ref(dst)} {val}"

  def sub_score(self, dst: Slot, val: int) -> str:
    return self.add_score(dst, -val)

  def op_score(
    self, dst: Slot, op: lowering.BinaryOp, src: Slot
  ) -> str:
    sym = ARITH_OPS[op]
    return (
      f"scoreboard players operation {self.score_ref(dst)} {sym} "
      f"{self.score_ref(src)}"
    )

  def if_score(
    self, lhs: Slot, op: CmpOp, rhs: Slot, neg: bool = False
  ) -> str:
    if op is CmpOp.EQ or op is CmpOp.NE:
      neg = neg != (op is CmpOp.NE)
      cond = ("if", "unless")[neg]
      return f"{cond} score {self.score_ref(lhs)} = {self.score_ref(rhs)}"

    cond = ("if", "unless")[neg]
    sym = CMP_SYMBOLS[op]
    return f"{cond} score {self.score_ref(lhs)} {sym} {self.score_ref(rhs)}"

  def if_true(self, slot: Slot) -> str:
    return f"unless score {self.score_ref(slot)} matches 0"

  def compare_score(
    self, dst: Slot, lhs: Slot, op: CmpOp, rhs: Slot
  ) -> str:
    return (
      f"execute store success score {self.score_ref(dst)} "
      f"{self.if_score(lhs, op, rhs)}"
    )

  # -- nbt -----------------------------------------------------------------
  def if_data(self, slot: Slot, lit: Lit, neg: bool = False) -> str:
    cond = ("if", "unless")[neg]
    return (
      f"{cond} data {self.container()} "
      f"{{{self.data_path(slot)}:{self.lit(lit)}}}"
    )

  def copy_data(self, dst: Slot, src: Slot) -> str:
    return f"data modify {self.data_ref(dst)} set from {self.data_ref(src)}"

  def set_data(self, dst: Slot, val: Lit) -> str:
    return f"data modify {self.data_ref(dst)} set value {self.lit(val)}"

  def equal_data(
    self, dst: Slot, lhs: Slot, rhs: Slot | Lit, eq: bool
  ) -> list[str]:
    """dst = (lhs == rhs), where values live in NBT."""
    if isinstance(rhs, Lit):
      return [
        f"execute store success score {self.score_ref(dst)} "
        f"{self.if_data(lhs, rhs, neg=not eq)}"
      ]

    # `data modify` reports failure when nothing changes, so a staged copy
    # that reports success means the two values differ.
    cmds = [
      self.copy_data(self._scratch, lhs),
      f"execute store success score {self.score_ref(dst)} run "
      + self.copy_data(self._scratch, rhs),
    ]
    if eq:
      cmds.append(
        f"execute store success score {self.score_ref(dst)} "
        f"if score {self.score_ref(dst)} matches 0"
      )
    return cmds

  # -- functions -----------------------------------------------------------
  def func_name(self, block: str) -> str:
    """The name a function value is stored under."""
    return f"{self.namespace}:{block}"

  def call(self, block: str) -> str:
    return f"function {self.func_name(block)}"

  def macro_call(self, block: str) -> str:
    """Call a macro function: the slot container provides its arguments."""
    return f"function {self.func_name(block)} with {self.container()}"

  # A function value is a macro argument: one trampoline per storage cell
  # expands to a call of whatever name that cell holds.  The cell's data path
  # doubles as the macro variable name.
  def dispatch_name(self, slot: Slot) -> str:
    return f".dispatch/{self.name(slot)}"

  def dispatch_body(self, slot: Slot) -> str:
    return f"$function $({self.name(slot)})"

  def dispatch_call(self, slot: Slot) -> str:
    return self.macro_call(self.dispatch_name(slot))

  # -- pointers ------------------------------------------------------------
  # A pointer value is the base name of a cell, so `$(key)` inside the
  # command text turns it into a cell reference: `#q.x` for a score, `q.x`
  # for a data path.
  def deref_read_score(self, key: str, path: str, dst: Slot) -> str:
    holder = f"#$(%s){path}" % key
    return (
      f"$scoreboard players operation {self.score_ref(dst)} = "
      f"{holder} {self.objective}"
    )

  def deref_read_data(self, key: str, path: str, dst: Slot) -> str:
    # cell names are flat keys, so the whole substituted path is one quoted
    # key (`s.name` would otherwise mean two nested keys)
    target = '"$(%s)%s"' % (key, path)
    return (
      f"$data modify {self.data_ref(dst)} set from {self.container()} "
      f"{target}"
    )

  def deref_write_score(self, key: str, path: str, src: Slot) -> str:
    holder = f"#$(%s){path}" % key
    return (
      f"$scoreboard players operation {holder} {self.objective} = "
      f"{self.score_ref(src)}"
    )

  def deref_write_data(self, key: str, path: str, src: Slot) -> str:
    target = '"$(%s)%s"' % (key, path)
    return (
      f"$data modify {self.container()} {target} set from "
      f"{self.data_ref(src)}"
    )

  def guard(self, cond: str, cmd: str) -> str:
    return f"execute {cond} run {cmd}"

  def jump(self, block: str, cond: str | None = None) -> str:
    cmd = f"return run {self.call(block)}"
    return cmd if cond is None else f"execute {cond} run {cmd}"

  def ret(self, val: int = 0) -> str:
    return f"return {val}"

  # -- setup ---------------------------------------------------------------
  def init_const(self, slot: Slot) -> str:
    lit = slot.const
    assert lit is not None

    if slot.store is Store.SCORE:
      return self.set_score(slot, lit.as_int())

    return self.set_data(slot, lit)

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
    self._counts: dict[str, int] = {}
    # NBT equality of two slots needs somewhere to stage a copy
    self._scratch = slots.alloc(Store.NBT)

  # -- naming --------------------------------------------------------------
  def _unique(self, hint: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.+\-]", "_", hint)
    n = self._counts.get(base, 0)
    self._counts[base] = n + 1
    return base if n == 0 else f"{base}{n}"

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

    name = self._unique(hint)

    if slot.store is Store.SCORE:
      name = ("$" if slot.export_name is not None else "#") + name

    self._names[slot] = name
    return name

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
      case lowering.ScalarType.STR:
        return json.dumps(lit.value, ensure_ascii=False)

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
  def copy_data(self, dst: Slot, src: Slot) -> str:
    return f"data modify {self.data_ref(dst)} set from {self.data_ref(src)}"

  def set_data(self, dst: Slot, val: Lit) -> str:
    return f"data modify {self.data_ref(dst)} set value {self.lit(val)}"

  def equal_data(
    self, dst: Slot, lhs: Slot, rhs: Slot | Lit, eq: bool
  ) -> list[str]:
    """dst = (lhs == rhs), where values live in NBT."""
    if isinstance(rhs, Lit):
      cond = ("if", "unless")[not eq]
      return [
        f"execute store success score {self.score_ref(dst)} {cond} data "
        f"{self.container()} {{{self.data_path(lhs)}:{self.lit(rhs)}}}"
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
  def call(self, block: str) -> str:
    return f"function {self.namespace}:{block}"

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

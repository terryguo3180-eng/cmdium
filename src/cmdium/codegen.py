from dataclasses import dataclass
from enum import auto

from cmdium import lowering
from cmdium import semantic
from cmdium import syntax
from cmdium.utils import DataEnum


class CmdVar:
  ...


@dataclass
class Slot(CmdVar):
  path: str
  kind: BaseType
  temp: bool = False


@dataclass
class Const(CmdVar):
  val: int
  kind: BaseType


def get_pathslot(place: lowering.Place):
  return Slot(place.path, kind_of(place.type), place.temp)

def get_immslot(imm: lowering.ImmValue):
  return Const(imm.val, kind_of(imm.type))


class BaseType(DataEnum):
  INT = auto()


def kind_of(type: semantic.TypeInfo) -> BaseType:
  if type == semantic.BuiltinTypes.i32:
    return BaseType.INT

  raise RuntimeError(f"no storage kind for {type!r}")


class DatapackGen:
  def __init__(self, insts: list[lowering.IR], ns: str):
    self.insts = insts
    self.ns = ns

    self.consts: set[int] = set()
    self.funcs: dict[str, list[str]] = {}

  def emit(self, cmd: str, func: str):
    self.funcs.setdefault(func, []).append(cmd)

  def const_holder(self, const: int) -> str:
    if const not in self.consts:
      raise RuntimeError(f"{const} is not registered")
    return f"%{const}"

  def holder(self, slot: CmdVar) -> str:
    if isinstance(slot, Const):
      return self.const_holder(slot.val)

    assert isinstance(slot, Slot)
    
    if slot.temp:
      return f"#{slot.path}"
    
    return f"${slot.path}"  # avoid conflict with real players

  def render(self) -> dict[str, list[str]]:
    self.emit(f"scoreboard objectives add {self.ns} dummy", func="setup")

    for inst in self.insts:
      self.gen_inst(inst)

    self.write_consts()
    return self.funcs

  def add_const(self, const: int):
    self.consts.add(const)

  def write_consts(self):
    for const in sorted(self.consts):
      self.emit(
        f"scoreboard players set {self.const_holder(const)} {self.ns} {const}",
        func="setup"
      )

  def gen_inst(self, inst: lowering.IR):
    match inst:
      case lowering.Copy():
        self.gen_copy(inst)

      case lowering.BinOp():
        self.gen_binop(inst)

      case _:
        raise RuntimeError(f"unhandled instruction: {inst!r}")

  def gen_copy(self, inst: lowering.Copy):
    match inst.src:
      case lowering.ImmValue():
        (dst,) = inst.dst.leaves()
        self.gen_copy_scalar(get_pathslot(dst), get_immslot(inst.src))

      case lowering.Place():
        for dst, src in zip(inst.dst.leaves(), inst.src.leaves()):
          self.gen_copy_scalar(get_pathslot(dst), get_pathslot(src))
          
      case _:
        raise RuntimeError(f"unhandled operand: {inst.src!r}")

  def gen_copy_scalar(self, dst: Slot, src: CmdVar):
    match dst, src:
      case (
        Slot(kind=BaseType.INT),
        Slot(kind=BaseType.INT),
      ):
        self.emit(
          f"scoreboard players operation "
          f"{self.holder(dst)} {self.ns} = {self.holder(src)} {self.ns}",
          func="main"
        )

      case (
        Slot(kind=BaseType.INT),
        Const(kind=BaseType.INT),
      ):
        self.emit(
          f"scoreboard players set {self.holder(dst)} {self.ns} {src.val}",
          func="main"
        )

      case _:
        raise RuntimeError(f"unhandled slots: {dst!r} and {src!r}")

  def gen_binop(self, inst: lowering.BinOp):
    match inst.dst, inst.bop, inst.src:
      case (
        lowering.Place(type=semantic.BuiltinTypes.i32) as dst,
        syntax.BinaryOp.ADD | syntax.BinaryOp.SUB as bop,
        lowering.ImmValue(type=semantic.BuiltinTypes.i32) as src,
      ):
        op = "add" if (bop == syntax.BinaryOp.ADD) == (src.val >= 0) else "remove"
        val = abs(src.val)

        dst_slot = get_pathslot(dst)
        
        self.emit(
          f"scoreboard players {op} {self.holder(dst_slot)} {self.ns} {val}",
          func="main"
        )

      case (
        lowering.Place(type=semantic.BuiltinTypes.i32) as dst,
        bop,
        lowering.Place(type=semantic.BuiltinTypes.i32)
        | lowering.ImmValue(type=semantic.BuiltinTypes.i32) as src,
      ):
        op = {
          syntax.BinaryOp.ADD: "+=",
          syntax.BinaryOp.SUB: "-=",
          syntax.BinaryOp.MUL: "*=",
          syntax.BinaryOp.DIV: "/=",
          syntax.BinaryOp.MOD: "%=",
        }[bop]

        if isinstance(src, lowering.ImmValue):
          self.add_const(src.val)
          src_holder = self.holder(get_immslot(src))
        else:
          src_holder = self.holder(get_pathslot(src))

        dst_slot = get_pathslot(dst)

        self.emit(
          f"scoreboard players operation "
          f"{self.holder(dst_slot)} {self.ns} {op} {src_holder} {self.ns}",
          func="main"
        )

      case _:
        raise RuntimeError(
          f"unhandled binops: {inst.dst!r}, {inst.bop!r}, {inst.src!r}"
        )

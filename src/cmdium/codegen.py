from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from cmdium import lowering
from cmdium.abi import ABI, CMP_OPS
from cmdium.slots import Lit, Slot, SlotTable, Store, Value, store_of


@dataclass
class MCFunction:
  name: str
  cmds: list[str]

  def emit(self, *cmds: str):
    self.cmds.extend(cmds)


class FuncGen:
  """Emits one mcfunction in terms of slots; it never spells out syntax."""

  def __init__(self, name: str, abi: ABI, slots: SlotTable, tmp: Slot):
    self.abi = abi
    self.slots = slots
    self.tmp = tmp
    self.func = MCFunction(name, [])

  def emit(self, *cmds: str):
    self.func.emit(*cmds)

  # -- operands ------------------------------------------------------------
  def value(self, op: lowering.Operand) -> Value:
    match op:
      case lowering.Var():
        return self.slots.of_var(op)

      case lowering.IntImm(val=val):
        return Lit(lowering.ScalarType.INT, val)

      case lowering.StrImm(val=val):
        return Lit(lowering.ScalarType.STR, val)

      case _:
        raise NotImplementedError(f"operand {op!r} as a value")

  def slot_of(self, op: Value) -> Slot:
    """Materialise a literal into a slot when a slot is required."""
    return op if isinstance(op, Slot) else self.slots.const(op)

  # -- instructions --------------------------------------------------------
  def copy(self, dst: Slot, src: Value):
    match dst.store, src:
      case Store.SCORE, Lit(type=lowering.ScalarType.INT):
        self.emit(self.abi.set_score(dst, src.as_int()))

      case Store.SCORE, Slot():
        self.emit(self.abi.copy_score(dst, src))

      case Store.NBT, Lit():
        self.emit(self.abi.set_data(dst, src))

      case Store.NBT, Slot():
        self.emit(self.abi.copy_data(dst, src))

      case _:
        raise NotImplementedError(f"cannot copy {src!r} into {dst!r}")

  def binop(
    self,
    dst: Slot,
    lhs: Value,
    op: lowering.BinaryOp,
    rhs: Value,
  ):
    if op in CMP_OPS:
      self.compare(dst, lhs, op, rhs)
    else:
      self.arith(dst, lhs, op, rhs)

  def compare(
    self,
    dst: Slot,
    lhs: Value,
    op: lowering.BinaryOp,
    rhs: Value,
  ):
    if lhs.store is Store.NBT or rhs.store is Store.NBT:
      if op not in (lowering.BinaryOp.CEQ, lowering.BinaryOp.CNE):
        raise NotImplementedError("ordering comparison on data values")

      # equality is symmetric, so keep a literal on the right-hand side
      if isinstance(lhs, Lit) and isinstance(rhs, Slot):
        lhs, rhs = rhs, lhs

      self.emit(*self.abi.equal_data(
        dst, self.slot_of(lhs), rhs, eq=op is lowering.BinaryOp.CEQ,
      ))
      return

    self.emit(self.abi.compare_score(
      dst, self.slot_of(lhs), CMP_OPS[op], self.slot_of(rhs),
    ))

  def arith(
    self,
    dst: Slot,
    lhs: Value,
    op: lowering.BinaryOp,
    rhs: Value,
  ):
    # dst may alias an operand; the commutative cases just swap, everything
    # else spills the old dst into a scratch slot first.
    if rhs == dst and lhs != dst:
      if op in (
        lowering.BinaryOp.ADD,
        lowering.BinaryOp.MUL,
        lowering.BinaryOp.OR,
        lowering.BinaryOp.AND,
      ):
        lhs, rhs = rhs, lhs
      else:
        self.emit(self.abi.copy_score(self.tmp, dst))
        rhs = self.tmp

    if lhs != dst:
      self.copy(dst, lhs)

    match op, rhs:
      case lowering.BinaryOp.ADD, Lit(type=lowering.ScalarType.INT):
        self.emit(self.abi.add_score(dst, rhs.as_int()))

      case lowering.BinaryOp.SUB, Lit(type=lowering.ScalarType.INT):
        self.emit(self.abi.sub_score(dst, rhs.as_int()))

      case _:
        self.emit(self.abi.op_score(dst, op, self.slot_of(rhs)))


class DatapackGen:
  def __init__(
    self, ir_funcs: list[lowering.IRFunc], ns: str, setup: str = ".setup"
  ):
    self.ir_funcs = ir_funcs
    self.ns = ns
    self.setup = setup

    # vars are unique objects, so the per-function tables can be merged
    exports: dict[lowering.Var, str] = {}
    for func in ir_funcs:
      exports.update(func.exports)

    self.slots = SlotTable(exports)
    self.abi = ABI(ns, ns, self.slots)

    self.tmp = self.slots.alloc(Store.SCORE)
    self.ret_slots: dict[tuple[lowering.IRFunc, int], Slot] = {}
    self.funcs: dict[str, MCFunction] = {}

  @contextmanager
  def with_func(self, name: str):
    func = FuncGen(name, self.abi, self.slots, self.tmp)
    try:
      yield func
    finally:
      self.funcs[name] = func.func

  def ret_slot(self, func: lowering.IRFunc, i: int) -> Slot:
    key = (func, i)
    slot = self.ret_slots.get(key)
    if slot is None:
      slot = self.slots.alloc(store_of(func.ret_types[i]))
      self.ret_slots[key] = slot
    return slot

  def gen(self) -> dict[str, MCFunction]:
    for func in self.ir_funcs:
      for block in func.blocks:
        with self.with_func(block.name) as fg:
          for inst in block.body:
            self.gen_inst(fg, inst)
          if block.term is not None:
            self.gen_term(fg, func, block.term)

    self.write_setup()
    return self.funcs

  def write_setup(self):
    with self.with_func(self.setup) as fg:
      fg.emit(self.abi.add_objective())
      for slot in self.slots.consts():
        fg.emit(self.abi.init_const(slot))

  def gen_inst(self, fg: FuncGen, inst: lowering.IR):
    match inst:
      case lowering.Copy(dst=dst, src=src):
        fg.copy(self.slots.of_var(dst), fg.value(src))

      case lowering.BinOp(dst=dst, lhs=lhs, bop=bop, rhs=rhs):
        fg.binop(
          self.slots.of_var(dst), fg.value(lhs), bop, fg.value(rhs)
        )

      case lowering.Call(callee=callee, args=args, rets=rets):
        func = callee.func
        for param, arg in zip(func.params, args, strict=True):
          fg.copy(self.slots.of_var(param), fg.value(arg))

        fg.emit(self.abi.call(func.entry.name))

        for i, ret in enumerate(rets):
          if ret is not None:
            fg.copy(self.slots.of_var(ret), self.ret_slot(func, i))

      case lowering.IndirectCall():
        raise NotImplementedError("indirect calls are not supported")

  def gen_term(self, fg: FuncGen, func: lowering.IRFunc, term: lowering.Terminator):
    match term:
      case lowering.Jump(target=target):
        fg.emit(self.abi.jump(target.bb.name))

      case lowering.Branch(cond=cond, then=then, orelse=orelse):
        fg.emit(
          self.abi.jump(then.bb.name, self.abi.if_true(self.slots.of_var(cond)))
        )
        fg.emit(self.abi.jump(orelse.bb.name))

      case lowering.Ret(values=values):
        for i, val in enumerate(values):
          fg.copy(self.ret_slot(func, i), fg.value(val))
        fg.emit(self.abi.ret())

      case lowering.IndirectJump():
        raise NotImplementedError("indirect jumps are not supported")

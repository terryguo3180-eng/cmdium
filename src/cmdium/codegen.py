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


# (macro variable, data path, pointee store, "read" | "write")
DerefKey = tuple[str, str, Store, str]


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

      case lowering.FuncImm(func=func):
        return func_lit(func)

      case lowering.PtrImm(val=val):
        return Lit(lowering.ScalarType.PTR, val)

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


@dataclass
class Frame:
  """Argument and result slots shared by all calls with one signature.

  Sharing by signature (rather than per function) is what lets an indirect
  call move values without knowing which function it is going to hit.
  """

  args: list[Slot]
  rets: list[Slot]


def func_lit(func: lowering.IRFunc) -> Lit:
  """The value stored for a function address."""
  return Lit(lowering.ScalarType.FUNC, func.entry.name)


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

    # An exported name is part of the pack's interface, so it is claimed
    # before any internal variable that happens to share the same hint.
    for var in exports:
      self.abi.name(self.slots.of_var(var))

    self.tmp = self.slots.alloc(Store.SCORE)
    self.funcs: dict[str, MCFunction] = {}
    self.frames: dict[lowering.Signature, Frame] = {}

    # target slot -> generated macro trampoline, one per cell, filled lazily
    self.dispatchers: dict[Slot, str] = {}
    self._dispatch_tmp: Slot | None = None

    # pointer trampolines: cache key -> function name, plus their bodies
    self._derefs: dict[DerefKey, str] = {}
    self._deref_bodies: dict[str, str] = {}
    self._deref_tmp: dict[Store, Slot] = {}
    self._pointer_tmp: Slot | None = None

  def dispatch(self, slot: Slot) -> str:
    """The trampoline that calls whatever function value `slot` holds."""
    name = self.dispatchers.get(slot)
    if name is None:
      name = self.abi.dispatch_name(slot)
      self.dispatchers[slot] = name
    return name

  def dispatch_slot(self, fg: FuncGen, slot: Slot) -> Slot:
    """The cell a macro can read the function value from.

    Macro variables must be plain words, but a cell like `pair.1` is a
    perfectly good NBT key.  Such a cell is staged into a scratch cell first.
    """
    if self.abi.is_macro_name(self.abi.name(slot)):
      return slot

    if self._dispatch_tmp is None:
      self._dispatch_tmp = self.slots.alloc(Store.NBT, hint="dispatch")
    fg.copy(self._dispatch_tmp, slot)
    return self._dispatch_tmp

  @contextmanager
  def with_func(self, name: str):
    func = FuncGen(name, self.abi, self.slots, self.tmp)
    try:
      yield func
    finally:
      self.funcs[name] = func.func

  def frame(self, sig: lowering.Signature) -> Frame:
    frame = self.frames.get(sig)
    if frame is None:
      frame = Frame(
        args=[self.slots.alloc(store_of(t)) for t in sig.params],
        rets=[self.slots.alloc(store_of(t)) for t in sig.rets],
      )
      self.frames[sig] = frame
    return frame

  def gen(self) -> dict[str, MCFunction]:
    for func in self.ir_funcs:
      frame = self.frame(func.signature)
      for block in func.blocks:
        with self.with_func(block.name) as fg:
          if block is func.entry:
            self.gen_prologue(fg, func, frame)

          for inst in block.body:
            self.gen_inst(fg, inst)
          if block.term is not None:
            self.gen_term(fg, func, block.term)

    self.write_dispatchers()
    self.write_derefs()
    self.write_setup()
    return self.funcs

  def write_derefs(self):
    for name, body in self._deref_bodies.items():
      with self.with_func(name) as fg:
        fg.emit(body)

  # -- pointers ------------------------------------------------------------
  def ptr_base(self, val: lowering.Val, path: str = "") -> str:
    """The name prefix shared by the cells of a pointer target."""
    match val:
      case lowering.Agg(fields=fields):
        if not fields:
          raise RuntimeError("an empty aggregate has no address")
        name, sub = fields[0]
        return self.ptr_base(sub, f"{path}.{name}" if path else name)

      case _:
        assert isinstance(val, lowering.Var), val
        base = self.abi.base(self.slots.of_var(val))
        if not path:
          return base

        suffix = "." + path
        if not base.endswith(suffix):
          raise RuntimeError(
            f"cannot locate the base of {base!r} for path {path!r}"
          )
        return base[: -len(suffix)]

  def pointer_tmp(self) -> Slot:
    """A cell a macro can read any pointer value from."""
    if self._pointer_tmp is None:
      self._pointer_tmp = self.slots.alloc(Store.NBT, hint="deref_ptr")
    return self._pointer_tmp

  def deref_key(self, fg: FuncGen, ptr_slot: Slot) -> str:
    """The macro variable that carries this pointer's value."""
    base = self.abi.base(ptr_slot)
    if self.abi.is_macro_name(base):
      return base

    tmp = self.pointer_tmp()
    fg.copy(tmp, ptr_slot)
    return self.abi.base(tmp)

  def deref_scratch(self, store: Store) -> Slot:
    slot = self._deref_tmp.get(store)
    if slot is None:
      slot = self.slots.alloc(store, hint="deref")
      self._deref_tmp[store] = slot
    return slot

  def gen_raw_macro(self, fg: FuncGen, src: lowering.Var):
    """`$cmd(<str value>)`: expand the cell into a command at run time."""
    key = self.deref_key(fg, self.slots.of_var(src))
    body = self.abi.raw_macro(key)
    name = self.deref_trampoline((key, "", Store.NBT, "raw"), body)
    fg.emit(self.abi.macro_call(name))

  def deref_trampoline(self, key: DerefKey, body: str) -> str:
    name = self._derefs.get(key)
    if name is None:
      name = f".deref/{len(self._derefs)}"
      self._derefs[key] = name
      self._deref_bodies[name] = body
    return name

  def gen_load_ptr(
    self, fg: FuncGen, dst: lowering.Var, ptr: lowering.Var,
    path: str, type: lowering.ScalarType,
  ):
    store = store_of(type)
    scratch = self.deref_scratch(store)
    key = self.deref_key(fg, self.slots.of_var(ptr))

    body = (
      self.abi.deref_read_score(key, path, scratch)
      if store is Store.SCORE
      else self.abi.deref_read_data(key, path, scratch)
    )
    name = self.deref_trampoline((key, path, store, "read"), body)

    fg.emit(self.abi.macro_call(name))
    fg.copy(self.slots.of_var(dst), scratch)

  def gen_store_ptr(
    self, fg: FuncGen, ptr: lowering.Var, path: str,
    type: lowering.ScalarType, src: lowering.Operand,
  ):
    store = store_of(type)
    scratch = self.deref_scratch(store)
    fg.copy(scratch, fg.value(src))

    key = self.deref_key(fg, self.slots.of_var(ptr))
    body = (
      self.abi.deref_write_score(key, path, scratch)
      if store is Store.SCORE
      else self.abi.deref_write_data(key, path, scratch)
    )
    name = self.deref_trampoline((key, path, store, "write"), body)

    fg.emit(self.abi.macro_call(name))

  def write_dispatchers(self):
    """One macro function per indirect-call target cell.

    `$function $(<cell>)` reads the name out of the slot container, so the
    target never has to be compared against a list of candidates.
    """
    for slot, name in self.dispatchers.items():
      with self.with_func(name) as fg:
        fg.emit(self.abi.dispatch_body(slot))

  def write_setup(self):
    with self.with_func(self.setup) as fg:
      fg.emit(self.abi.add_objective())
      for slot in self.slots.consts():
        fg.emit(self.abi.init_const(slot))

  def gen_prologue(
    self, fg: FuncGen, func: lowering.IRFunc, frame: Frame
  ):
    """Move the shared arguments into this function's own parameters.

    Without the copy, a same-signature call made by the callee would clobber
    the parameters of its caller.
    """
    for param, src in zip(func.params, frame.args, strict=True):
      fg.copy(self.slots.of_var(param), src)

  def gen_inst(self, fg: FuncGen, inst: lowering.IR):
    match inst:
      case lowering.Copy(dst=dst, src=src):
        fg.copy(self.slots.of_var(dst), fg.value(src))

      case lowering.BinOp(dst=dst, lhs=lhs, bop=bop, rhs=rhs):
        fg.binop(
          self.slots.of_var(dst), fg.value(lhs), bop, fg.value(rhs)
        )

      case lowering.Call(callee=callee, args=args, rets=rets):
        self.gen_call(fg, callee.func, args, rets)

      case lowering.IndirectCall(
        target=target, args=args, rets=rets, signature=sig
      ):
        self.gen_indirect_call(fg, target, args, rets, sig)

      case lowering.AddrOf(dst=dst, target=target):
        fg.emit(self.abi.set_data(
          self.slots.of_var(dst),
          Lit(lowering.ScalarType.PTR, self.ptr_base(target)),
        ))

      case lowering.LoadPtr(dst=dst, ptr=ptr, path=path, type=type):
        self.gen_load_ptr(fg, dst, ptr, path, type)

      case lowering.StorePtr(ptr=ptr, path=path, type=type, src=src):
        self.gen_store_ptr(fg, ptr, path, type, src)

      case lowering.Raw(cmd=cmd):
        fg.emit(cmd)

      case lowering.RawMacro(src=src):
        self.gen_raw_macro(fg, src)

  def gen_call(
    self,
    fg: FuncGen,
    func: lowering.IRFunc,
    args: tuple[lowering.Operand, ...],
    rets: tuple[lowering.Var | None, ...],
  ):
    frame = self.frame(func.signature)
    self.gen_args(fg, frame, args)
    fg.emit(self.abi.call(func.entry.name))
    self.gen_rets(fg, frame, rets)

  def gen_indirect_call(
    self,
    fg: FuncGen,
    target: lowering.Var,
    args: tuple[lowering.Operand, ...],
    rets: tuple[lowering.Var | None, ...],
    sig: lowering.Signature,
  ):
    frame = self.frame(sig)
    self.gen_args(fg, frame, args)

    # a macro line expands the function value held in the cell into a call,
    # which is cheaper than testing the value against every candidate
    slot = self.dispatch_slot(fg, self.slots.of_var(target))
    self.dispatch(slot)
    fg.emit(self.abi.dispatch_call(slot))

    self.gen_rets(fg, frame, rets)

  def gen_args(
    self,
    fg: FuncGen,
    frame: Frame,
    args: tuple[lowering.Operand, ...],
  ):
    for slot, arg in zip(frame.args, args, strict=True):
      fg.copy(slot, fg.value(arg))

  def gen_rets(
    self,
    fg: FuncGen,
    frame: Frame,
    rets: tuple[lowering.Var | None, ...],
  ):
    for slot, ret in zip(frame.rets, rets, strict=True):
      if ret is not None:
        fg.copy(self.slots.of_var(ret), slot)

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
        frame = self.frame(func.signature)
        for slot, val in zip(frame.rets, values, strict=True):
          fg.copy(slot, fg.value(val))
        fg.emit(self.abi.ret())

      case lowering.IndirectJump():
        raise NotImplementedError("indirect jumps are not supported")

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto
from typing import cast

from cmdium import semantic
from cmdium import syntax
from cmdium.utils import DataEnum


class ScalarType(DataEnum):
  INT = auto()
  LABEL = auto()


@dataclass(frozen=True)
class Scalar:
  type: ScalarType


@dataclass(frozen=True)
class Aggregate:
  fields: tuple[tuple[str, Layout], ...]


Layout = Scalar | Aggregate


def layout_of(type: semantic.TypeInfo) -> Layout:
  match type:
    case semantic.Struct(fields=fields):
      return Aggregate(tuple((n, layout_of(t)) for n, t in fields.items()))

    case semantic.Prim(kind=kind):
      match kind:
        case semantic.PrimKind.I32 | semantic.PrimKind.BOOL:
          return Scalar(ScalarType.INT)

        case semantic.PrimKind.VOID:
          return Aggregate(())

    case semantic.Function():
      raise NotImplementedError("function values have no layout yet")


@dataclass(frozen=True)
class Place:
  path: str
  layout: Layout
  temp: bool = False

  def project(self, name: str) -> Place:
    assert isinstance(self.layout, Aggregate)
    for n, lay in self.layout.fields:
      if n == name:
        return Place(f"{self.path}.{name}", lay, self.temp)
    raise RuntimeError(f"unknown field: {name!r}")

  def leaves(self) -> Iterator[Place]:
    match self.layout:
      case Aggregate(fields=fields):
        for name, _ in fields:
          yield from self.project(name).leaves()

      case Scalar():
        yield self

  @property
  def layout_type(self) -> ScalarType:
    assert isinstance(self.layout, Scalar)
    return self.layout.type

  def __str__(self):
    return self.path


@dataclass(frozen=True)
class ImmValue:
  val: str
  kind: ScalarType

  @property
  def layout(self) -> Layout:
    return Scalar(self.kind)

  def __str__(self):
    return self.val


Operand = Place | ImmValue


class IRBase:
  def emit_leaves(self) -> Iterator[IR]:
    yield cast("IR", self)


@dataclass
class Copy(IRBase):
  dst: Place
  src: Operand

  def emit_leaves(self) -> Iterator[IR]:
    assert self.dst.layout == self.src.layout, (
      f"layout mismatch: {self.dst.layout!r} vs {self.src.layout!r}"
    )

    match self.src:
      case Place():
        for d, s in zip(self.dst.leaves(), self.src.leaves()):
          yield Copy(d, s)

      case ImmValue():
        yield self

  def __str__(self):
    return f"{self.dst!s} = {self.src!s}"


@dataclass
class SetZero(IRBase):
  dst: Place

  def emit_leaves(self) -> Iterator[IR]:
    for leaf in self.dst.leaves():
      assert isinstance(leaf.layout, Scalar)
      yield Copy(leaf, ImmValue("0", leaf.layout.type))

  def __str__(self):
    return f"zero {self.dst!s}"


class BinaryOp(DataEnum):
  ADD = auto()  # +
  SUB = auto()  # -
  MUL = auto()  # *
  DIV = auto()  # /
  MOD = auto()  # %

  OR = auto()   # |
  AND = auto()  # &

  CLT = auto()  # <
  CGT = auto()  # >
  CLE = auto()  # <=
  CGE = auto()  # >=
  CEQ = auto()  # ==
  CNE = auto()  # !=


def binaryop_of(binop: syntax.BinaryOp) -> BinaryOp:
  match binop:
    case syntax.BinaryOp.ADD: return BinaryOp.ADD
    case syntax.BinaryOp.SUB: return BinaryOp.SUB
    case syntax.BinaryOp.MUL: return BinaryOp.MUL
    case syntax.BinaryOp.DIV: return BinaryOp.DIV
    case syntax.BinaryOp.MOD: return BinaryOp.MOD
    case syntax.BinaryOp.OR: return BinaryOp.OR
    case syntax.BinaryOp.AND: return BinaryOp.AND
    case syntax.BinaryOp.CLT: return BinaryOp.CLT
    case syntax.BinaryOp.CGT: return BinaryOp.CGT
    case syntax.BinaryOp.CLE: return BinaryOp.CLE
    case syntax.BinaryOp.CGE: return BinaryOp.CGE
    case syntax.BinaryOp.CEQ: return BinaryOp.CEQ
    case syntax.BinaryOp.CNE: return BinaryOp.CNE


@dataclass
class BinOp(IRBase):
  dst: Place
  lhs: Operand
  bop: BinaryOp
  rhs: Operand

  def __str__(self):
    return f"{self.dst!s} = {self.lhs!s} {self.bop.name.lower()} {self.rhs!s}"


@dataclass
class Jump:
  target: BasicBlock

  def successors(self) -> Iterator[BasicBlock]:
    yield self.target

  def __str__(self):
    return f"jump {self.target.addr!s}"


@dataclass
class IndirectJump:
  target: Place  # LABEL
  possible: tuple[BasicBlock, ...]

  def __post_init__(self):
    assert self.target.layout == Scalar(ScalarType.LABEL)

  def successors(self) -> Iterator[BasicBlock]:
    yield from self.possible

  def __str__(self):
    return f"jump* {self.target!s}"


@dataclass
class Branch:
  cond: Place
  then: BasicBlock
  orelse: BasicBlock

  def __post_init__(self):
    assert self.cond.layout == Scalar(ScalarType.INT)

  def successors(self) -> Iterator[BasicBlock]:
    yield self.then
    yield self.orelse

  def __str__(self):
    return f"br {self.cond!s} ? {self.then.addr!s} : {self.orelse.addr!s}"


@dataclass
class Ret:
  def successors(self) -> Iterator[BasicBlock]:
    return; yield

  def __str__(self):
    return f"ret"


@dataclass(eq=False)
class BasicBlock:
  addr: ImmValue
  body: list[IR]
  term: Terminator | None = None

  def emit(self, inst: IR):
    self.body.append(inst)

  def scalarize(self):
    self.body = [leaf for inst in self.body for leaf in inst.emit_leaves()]

  def __str__(self):
    s = f"{self.addr!s}:\n"
    for inst in self.body:
      s += f"  {inst!s}\n"
    if self.term is not None:
      s += f"  {self.term!s}\n"
    return s.strip()


@dataclass(eq=False)
class IRFunc:
  name: str
  params: list[Place]
  ret: Place
  blocks: list[BasicBlock]

  @property
  def entry(self) -> BasicBlock:
    return self.blocks[0]

  @property
  def addr(self) -> Operand:
    return self.entry.addr

  def __str__(self):
    s = f"func {self.name} {{\n"
    for inst in self.blocks:
      for line in str(inst).split("\n"):
        s += "  " + str(line) + "\n"
      s += "\n"
    return s.strip() + "\n}"


@dataclass
class Call(IRBase):
  callee: IRFunc

  def __str__(self):
    return f"call {self.callee.name}"


@dataclass
class IndirectCall(IRBase):
  target: Place  # LABEL
  possible: tuple[IRFunc, ...]


IR = Copy | SetZero | BinOp | Call | IndirectCall
Terminator = Jump | IndirectJump | Branch | Ret


class FuncBuilder:
  def __init__(self, name: str, ret_layout: Layout):
    self.name = name
    self.ret = Place(f"{name}.ret", ret_layout)
    self.params: list[Place] = []
    self.blocks: list[BasicBlock] = []
    self.func = IRFunc(name, self.params, self.ret, self.blocks)

    self.break_stack: list[BasicBlock] = []
    self.continue_stack: list[BasicBlock] = []

    self.n_blocks = 0
    self.n_slots = 0
    self.cur = self.start(self.new_block("entry"))

  def new_block(self, hint: str = "bb") -> BasicBlock:
    self.n_blocks += 1
    addr = ImmValue(f"{self.name}.{hint}{self.n_blocks}", ScalarType.LABEL)
    return BasicBlock(addr, [])

  def param(self, name: str, layout: Layout) -> Place:
    p = self.var(name, layout)
    self.params.append(p)
    return p

  def start(self, block: BasicBlock) -> BasicBlock:
    assert not self.blocks or self.cur.term is not None
    self.blocks.append(block)
    self.cur = block
    return block

  @property
  def open(self) -> bool:
    return self.cur.term is None

  def as_place(self, op: Operand) -> Place:
    match op:
      case Place():
        return op
      
      case ImmValue():
        tmp = self.temp(op.layout)
        self.emit(Copy(tmp, op))
        return tmp

  def emit(self, inst: IR):
    assert self.open
    self.cur.body.append(inst)

  def set_term(self, term: Terminator):
    assert self.open
    self.cur.term = term

  def goto(self, target: BasicBlock):
    if self.open:
      self.set_term(Jump(target))

  def branch(self, cond: Operand, then: BasicBlock, orelse: BasicBlock):
    match cond:
      case ImmValue(val=v):
        self.goto(then if int(v) else orelse)

      case _:
        self.set_term(Branch(self.as_place(cond), then, orelse))

  @contextmanager
  def push_loop(self, break_b: BasicBlock, continue_b: BasicBlock):
    self.break_stack.append(break_b)
    self.continue_stack.append(continue_b)
    try:
      yield
    finally:
      self.break_stack.pop()
      self.continue_stack.pop()

  def break_block(self):
    return self.break_stack[-1]

  def continue_block(self):
    return self.continue_stack[-1]

  def var(self, name: str, layout: Layout) -> Place:
    self.n_slots += 1
    return Place(f"{self.name}.{name}{self.n_slots}", layout)

  def temp(self, layout: Layout) -> Place:
    self.n_slots += 1
    return Place(f"{self.name}.t{self.n_slots}", layout, temp=True)

  def _check_cfg(self):
    ids = {id(b) for b in self.func.blocks}
    for b in self.func.blocks:
      assert b.term is not None, f"{b.addr!s}: no terminator"
      for s in b.term.successors():
        assert id(s) in ids, \
          f"{b.addr!s} -> {s.addr!s}: successor block not inside current func"

  def finish(self) -> IRFunc:
    if self.open:
      self.set_term(Ret())
    self._check_cfg()
    return self.func


class Lowering:
  def __init__(self, tree: syntax.CompilationUnit):
    self.tree = tree
    self.fb = FuncBuilder(".main", Scalar(ScalarType.INT))
    self.funcs: dict[semantic.Symbol, IRFunc] = {}
    self.builders: dict[semantic.Symbol, FuncBuilder] = {}
    self.sym_places: dict[semantic.Symbol, Place] = {}

  def declare_func(self, node: syntax.FuncDecl):
    fty = node.sym.type
    assert isinstance(fty, semantic.Function)
    fb = FuncBuilder(node.name, layout_of(fty.ret_type))
    
    for p in node.params:
      self.sym_places[p.sym] = fb.param(p.name, layout_of(p.sym.type))
  
    self.funcs[node.sym] = fb.func
    self.builders[node.sym] = fb

  def lower(self) -> list[IRFunc]:
    main = self.fb
    for stmt in self.tree.body:
      if isinstance(stmt, syntax.FuncDecl):
        self.declare_func(stmt)

    for stmt in self.tree.body:
      self.lower_stmt(stmt)

    return [main.finish(), *(b.finish() for b in self.builders.values())]

  def lower_stmt(self, node: syntax.Statement):
    match node:
      case syntax.FuncDecl(name=name, body=body, sym=sym):
        if node.sym not in self.funcs:
          self.declare_func(node)
        
        prev, self.fb = self.fb, self.builders[sym]
        self.lower_stmt(body)
        self.fb = prev

      case syntax.VarDecl(name=name, init=init, sym=sym):
        if init is not None:
          ival = self.lower_expr(init)

        dst = self.fb.var(name, layout_of(sym.type))
        self.sym_places[sym] = dst

        if init is None:
          self.fb.emit(SetZero(dst))
        else:
          self.fb.emit(Copy(dst, ival))  # type: ignore

      case syntax.Assign(dst=dst, val=val):
        dst_v = self.lower_expr(dst)
        val_v = self.lower_expr(val)

        assert isinstance(dst_v, Place)
        self.fb.emit(Copy(dst_v, val_v))

      case syntax.AugAssign(dst=dst, val=val, bop=bop_node):
        dst_v = self.lower_expr(dst)
        val_v = self.lower_expr(val)

        assert isinstance(dst_v, Place)
        self.fb.emit(BinOp(dst_v, dst_v, binaryop_of(bop_node.bop), val_v))

      case syntax.ExprStmt(val=val):
        self.lower_expr(val)

      case syntax.Block(body=body):
        for stmt in body:
          self.lower_stmt(stmt)

      case syntax.If(test=test, body=body, orelse=orelse):
        cond = self.lower_expr(test)
        then_b = self.fb.new_block("then")
        end_b = self.fb.new_block("end")
        else_b = self.fb.new_block("else") if orelse is not None else end_b
        self.fb.branch(cond, then_b, else_b)

        self.fb.start(then_b)
        self.lower_stmt(body)
        self.fb.goto(end_b)

        if orelse is not None:
          self.fb.start(else_b)
          self.lower_stmt(orelse)
          self.fb.goto(end_b)

        self.fb.start(end_b)

      case syntax.While(test=test, body=body):
        head_b, body_b, end_b = (
          self.fb.new_block(n) for n in ("cond", "body", "end")
        )
        self.fb.goto(head_b)
        self.fb.start(head_b)
        if test is None:
          self.fb.goto(body_b)
        else:
          self.fb.branch(self.lower_expr(test), body_b, end_b)

        with self.fb.push_loop(end_b, head_b):
          self.fb.start(body_b)
          self.lower_stmt(body)
        
        self.fb.goto(head_b)
        self.fb.start(end_b)

      case syntax.Break():
        self.fb.goto(self.fb.break_block())
        self.fb.start(self.fb.new_block("dead"))

      case syntax.Continue():
        self.fb.goto(self.fb.continue_block())
        self.fb.start(self.fb.new_block("dead"))

      case syntax.Return(val=val):
        if val is not None:
          v = self.lower_expr(val)
          self.fb.emit(Copy(self.fb.ret, v))
        self.fb.set_term(Ret())
        self.fb.start(self.fb.new_block("dead"))

      case syntax.Pass():
        pass

      case syntax.TypeDecl() | syntax.StructDecl():
        pass

  def lower_expr(self, node: syntax.Expression) -> Operand:
    match node:
      case syntax.BinaryExpr(lhs=lhs, rhs=rhs, bop=bop_node):
        dst_v = self.fb.temp(layout_of(node.type))
        lhs_v = self.lower_expr(lhs)
        rhs_v = self.lower_expr(rhs)
        self.fb.emit(BinOp(dst_v, lhs_v, binaryop_of(bop_node.bop), rhs_v))

        return dst_v

      case syntax.UnaryExpr(val=val, uop=uop_node):
        v = self.lower_expr(val)
        dst = self.fb.temp(layout_of(node.type))
        zero = ImmValue("0", ScalarType.INT)

        match uop_node.uop:
          case syntax.UnaryOp.NEG:
            self.fb.emit(BinOp(dst, zero, BinaryOp.SUB, v))
          case syntax.UnaryOp.NOT:
            self.fb.emit(BinOp(dst, v, BinaryOp.CEQ, zero))

        return dst

      case syntax.Call(src=src, args=args):
        if not isinstance(src, syntax.Name):
          raise NotImplementedError("indirect call not supported yet")
        callee = self.funcs[src.sym]

        vals = [self.lower_expr(a) for a in args]
        for param, v in zip(callee.params, vals):
          self.fb.emit(Copy(param, v))

        self.fb.emit(Call(callee))

        res = self.fb.temp(callee.ret.layout)
        self.fb.emit(Copy(res, callee.ret))
        return res

      case syntax.Name() | syntax.MemberAccess():
        return self.lower_lval(node)

      case syntax.ScalarLiteral(text=text, type=type):
        layout = layout_of(type)

        if type == semantic.Prim(semantic.PrimKind.BOOL):
          text = "10"["TF".index(text)]

        assert isinstance(layout, Scalar)
        return ImmValue(text, layout.type)

  def lower_lval(self, node: syntax.Name | syntax.MemberAccess) -> Place:
    match node:
      case syntax.Name(sym=sym):
        return self.sym_places[sym]

      case syntax.MemberAccess(src=src, att=att):
        src_v = self.lower_expr(src)
        assert isinstance(src_v, Place)
        return src_v.project(att)

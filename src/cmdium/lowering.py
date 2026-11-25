from __future__ import annotations

import json

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto

from cmdium import semantic
from cmdium import syntax
from cmdium import utils


class ScalarType(utils.DataEnum):
  INT = auto()
  LABEL = auto()
  FUNC = auto()
  STR = auto()


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
      return Aggregate(
        tuple((n, layout_of(t)) for n, t in fields.items())
      )

    case semantic.Prim(kind=kind):
      match kind:
        case semantic.PrimKind.I32 | semantic.PrimKind.BOOL:
          return Scalar(ScalarType.INT)

        case semantic.PrimKind.STR:
          return Scalar(ScalarType.STR)

    case semantic.Tuple(fields=fields):
      return Aggregate(
        tuple((str(i), layout_of(t)) for i, t in enumerate(fields))
      )

    case semantic.Function():
      raise NotImplementedError("function values have no layout yet")


@dataclass(eq=False)
class Var:
  type: ScalarType
  hint: str
  id: int

  def __str__(self):
    return f"%{self.hint}.{self.id}"


@dataclass(frozen=True)
class IntImm:
  val: int

  @property
  def type(self) -> ScalarType:
    return ScalarType.INT

  def __str__(self):
    return str(self.val)


@dataclass(frozen=True)
class LabelImm:
  bb: BasicBlock

  @property
  def type(self) -> ScalarType:
    return ScalarType.LABEL

  def __str__(self):
    return self.bb.name


@dataclass(frozen=True)
class FuncImm:
  func: IRFunc

  @property
  def type(self) -> ScalarType:
    return ScalarType.FUNC

  def __str__(self):
    return self.func.name


@dataclass(frozen=True)
class StrImm:
  val: str

  @property
  def type(self) -> ScalarType:
    return ScalarType.STR

  def __str__(self):
    return json.dumps(self.val, ensure_ascii=False)


ImmValue = IntImm | LabelImm | FuncImm | StrImm
Operand = Var | ImmValue


@dataclass
class Copy:
  dst: Var
  src: Operand

  def __post_init__(self):
    assert self.dst.type == self.src.type, (
      f"type mismatch: {self.dst.type!r} vs {self.src.type!r}"
    )

  def __str__(self):
    return f"{self.dst!s} = {self.src!s}"


def zero_of(type: ScalarType) -> ImmValue:
  match type:
    case ScalarType.INT:
      return IntImm(0)

    case ScalarType.LABEL:
      # '-.-', never used but valid
      return LabelImm(BasicBlock("-.-", []))

    case ScalarType.FUNC:
      return FuncImm(IRFunc("-.-", [], [], []))

    case ScalarType.STR:
      return StrImm("")


class BinaryOp(utils.DataEnum):
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
class BinOp:
  dst: Var
  lhs: Operand
  bop: BinaryOp
  rhs: Operand

  def __str__(self):
    return f"{self.dst!s} = {self.lhs!s} {self.bop.name.lower()} {self.rhs!s}"


@dataclass
class Jump:
  target: LabelImm

  def successors(self) -> Iterator[BasicBlock]:
    yield self.target.bb

  def __str__(self):
    return f"jump {self.target!s}"


@dataclass
class IndirectJump:
  target: Var  # LABEL
  possible: tuple[BasicBlock, ...]

  def __post_init__(self):
    assert self.target.type == ScalarType.LABEL

  def successors(self) -> Iterator[BasicBlock]:
    yield from self.possible

  def __str__(self):
    return f"jump* {self.target!s}"


@dataclass
class Branch:
  cond: Var
  then: LabelImm
  orelse: LabelImm

  def __post_init__(self):
    assert self.cond.type == ScalarType.INT

  def successors(self) -> Iterator[BasicBlock]:
    yield self.then.bb
    yield self.orelse.bb

  def __str__(self):
    return f"br {self.cond!s} ? {self.then!s} : {self.orelse!s}"


@dataclass
class Ret:
  values: tuple[Operand, ...] = ()

  def successors(self) -> Iterator[BasicBlock]:
    return; yield

  def __str__(self):
    if not self.values:
      return "ret"
    return "ret " + ", ".join(str(v) for v in self.values)


@dataclass(eq=False)
class BasicBlock:
  name: str
  body: list[IR]
  term: Terminator | None = None

  def emit(self, inst: IR):
    self.body.append(inst)

  def __str__(self):
    s = f"{self.name}:\n"
    for inst in self.body:
      s += f"  {inst!s}\n"
    if self.term is not None:
      s += f"  {self.term!s}\n"
    return s.strip()


@dataclass(eq=False)
class IRFunc:
  name: str
  params: list[Var]
  ret_types: list[ScalarType]
  blocks: list[BasicBlock]
  # vars that are exposed outside the program, mapped to their public name
  exports: dict[Var, str] = field(default_factory=dict[Var, str])

  @property
  def entry(self) -> BasicBlock:
    return self.blocks[0]

  def __str__(self):
    params = ", ".join(str(p) for p in self.params)
    s = f"func {self.name}({params}) {{\n"
    for var, name in self.exports.items():
      s += f"  export {var!s} as {name}\n"
    for inst in self.blocks:
      for line in str(inst).split("\n"):
        s += "  " + str(line) + "\n"
      s += "\n"
    return s.strip() + "\n}"


@dataclass
class Call:
  callee: FuncImm
  args: tuple[Operand, ...]
  rets: tuple[Var | None, ...]  # None: result is unused

  def __str__(self):
    args = ", ".join(str(a) for a in self.args)
    call = f"call {self.callee!s}({args})"
    if all(r is None for r in self.rets):
      return call

    rets = ", ".join("_" if r is None else str(r) for r in self.rets)
    return f"{rets} = {call}"


@dataclass
class IndirectCall:
  target: Var  # LABEL
  possible: tuple[IRFunc, ...]


IR = Copy | BinOp | Call | IndirectCall
Terminator = Jump | IndirectJump | Branch | Ret


@dataclass(frozen=True)
class Agg:
  fields: tuple[tuple[str, Val], ...]

  def get(self, name: str) -> Val:
    for n, v in self.fields:
      if n == name:
        return v
    raise RuntimeError(f"unknown field: {name!r}")


# lowering-time value, IR itself only sees the scalar leaves
Val = Operand | Agg


def build(
  layout: Layout,
  leaf: Callable[[str, ScalarType], Operand],
  path: str = "",
) -> Val:
  match layout:
    case Aggregate(fields=fields):
      return Agg(tuple(
        (name, build(lay, leaf, f"{path}.{name}")) for name, lay in fields
      ))

    case Scalar(type=type):
      return leaf(path, type)


def leaves(val: Val) -> Iterator[Operand]:
  match val:
    case Agg(fields=fields):
      for _, v in fields:
        yield from leaves(v)

    case _:
      yield val


def leaf_types(layout: Layout) -> list[ScalarType]:
  match layout:
    case Aggregate(fields=fields):
      types: list[ScalarType] = []
      for _, lay in fields:
        types.extend(leaf_types(lay))
      return types

    case Scalar(type=type):
      return [type]


def zero_val(layout: Layout) -> Val:
  return build(layout, lambda _, type: zero_of(type))


class FuncBuilder:
  def __init__(self, name: str, ret_layout: Layout):
    self.name = name
    self.ret_layout = ret_layout
    self.params: list[Var] = []
    self.blocks: list[BasicBlock] = []
    self.func = IRFunc(name, self.params, leaf_types(ret_layout), self.blocks)

    self.break_stack: list[BasicBlock] = []
    self.continue_stack: list[BasicBlock] = []

    self.n_blocks = 0
    self.n_vars = 0
    self.cur = self.start(self.new_block("entry"))

  def new_block(self, hint: str) -> BasicBlock:
    self.n_blocks += 1
    return BasicBlock(f"{self.name}/{hint}{self.n_blocks}", [])

  def param(self, name: str, layout: Layout) -> Val:
    val = self.fresh(layout, name)
    for leaf in leaves(val):
      assert isinstance(leaf, Var)
      self.params.append(leaf)
    return val

  def start(self, block: BasicBlock) -> BasicBlock:
    assert not self.blocks or self.cur.term is not None
    self.blocks.append(block)
    self.cur = block
    return block

  @property
  def open(self) -> bool:
    return self.cur.term is None

  def emit(self, inst: IR):
    assert self.open
    self.cur.body.append(inst)

  def set_term(self, term: Terminator):
    assert self.open
    self.cur.term = term

  def goto(self, target: BasicBlock):
    if self.open:
      self.set_term(Jump(LabelImm(target)))

  def branch(self, cond: Operand, then: BasicBlock, orelse: BasicBlock):
    match cond:
      case IntImm(val=v):
        self.goto(then if v else orelse)

      case LabelImm() | FuncImm() | StrImm():
        raise NotImplementedError

      case Var():
        self.set_term(Branch(cond, LabelImm(then), LabelImm(orelse)))

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

  def new_var(self, type: ScalarType, hint: str) -> Var:
    self.n_vars += 1
    return Var(type, hint, self.n_vars)

  def temp(self, type: ScalarType) -> Var:
    return self.new_var(type, "t")

  def fresh(self, layout: Layout, hint: str) -> Val:
    return build(layout, lambda path, type: self.new_var(type, hint + path))

  def export(self, layout: Layout, val: Val, name: str) -> None:
    """Expose every scalar leaf of `val` under `name`.

    Aggregate fields get a dotted suffix (e.g. ``q.p.x``), matching the paths
    built by `fresh`.
    """
    match layout, val:
      case Aggregate(fields=fields), Agg(fields=vals):
        for (fname, lay), (_, sub) in zip(fields, vals, strict=True):
          self.export(lay, sub, f"{name}.{fname}")

      case Scalar(), Var() as var:
        self.func.exports[var] = name

      case _:
        raise RuntimeError("value shape mismatch")

  def copy_into(self, dst: Val, src: Val):
    match dst, src:
      case Agg(fields=dfields), Agg(fields=sfields):
        assert [n for n, _ in dfields] == [n for n, _ in sfields]
        for (_, d), (_, s) in zip(dfields, sfields):
          self.copy_into(d, s)

      case Var(), Var() | IntImm() | LabelImm() | FuncImm() | StrImm():
        self.emit(Copy(dst, src))

      case _:
        raise RuntimeError("value shape mismatch")

  def _check_cfg(self):
    ids = {id(b) for b in self.func.blocks}
    for b in self.func.blocks:
      assert b.term is not None, f"{b.name}: no terminator"
      for s in b.term.successors():
        assert id(s) in ids, \
          f"{b.name!s} -> {s!s}: successor block not inside current func"

      if isinstance(b.term, Ret):
        types = [v.type for v in b.term.values]
        assert types == self.func.ret_types, (
          f"{b.name}: returns {types!r}, expected {self.func.ret_types!r}"
        )

  def finish(self) -> IRFunc:
    if self.open:
      self.set_term(Ret(tuple(leaves(zero_val(self.ret_layout)))))
    self._check_cfg()
    return self.func


class Lowering:
  def __init__(self, tree: syntax.CompilationUnit, entry_name: str):
    self.tree = tree
    self.fb = FuncBuilder(entry_name, Aggregate(()))
    self.funcs: dict[semantic.Symbol, IRFunc] = {}
    self.builders: dict[semantic.Symbol, FuncBuilder] = {}
    self.sym_vals: dict[semantic.Symbol, Val] = {}

  def declare_func(self, node: syntax.FuncDecl):
    fty = node.sym.type
    assert isinstance(fty, semantic.Function)
    fb = FuncBuilder(node.name, layout_of(fty.ret_type))
    
    for p in node.params:
      self.sym_vals[p.sym] = fb.param(p.name, layout_of(p.sym.type))
  
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
    attrs = node.attrs

    match node:
      case syntax.FuncDecl(name=name, body=body, sym=sym):
        if node.sym not in self.funcs:
          self.declare_func(node)
        
        prev, self.fb = self.fb, self.builders[sym]
        self.lower_stmt(body)
        self.fb = prev

      case syntax.VarDecl(name=name, init=init, sym=sym):
        layout = layout_of(sym.type)
        if init is None:
          ival = zero_val(layout)
        else:
          ival = self.lower_expr(init)

        dst = self.fb.fresh(layout, name)
        self.sym_vals[sym] = dst
        self.fb.copy_into(dst, ival)

        for attr in attrs:
          match attr:
            case semantic.Export(hint=hint):
              self.fb.export(layout, dst, hint or name)

            case _:
              raise NotImplementedError  # handled in semantic

      case syntax.Assign(dst=dst, val=val):
        dst_v = self.lower_expr(dst)
        val_v = self.lower_expr(val)

        # the elements may read dst, e.g. `t = (t.1, t.0);`
        if isinstance(val, (syntax.TupleLiteral, syntax.StructLiteral)):
          tmp = self.fb.fresh(layout_of(val.type), "t")
          self.fb.copy_into(tmp, val_v)
          val_v = tmp

        self.fb.copy_into(dst_v, val_v)

      case syntax.AugAssign(dst=dst, val=val, bop=bop_node):
        dst_v = self.lower_scalar(dst)
        val_v = self.lower_scalar(val)

        assert isinstance(dst_v, Var)
        self.fb.emit(BinOp(dst_v, dst_v, binaryop_of(bop_node.bop), val_v))

      case syntax.ExprStmt(val=val):
        self.lower_expr(val)

      case syntax.Block(body=body):
        for s in body:
          if isinstance(s, syntax.FuncDecl) and s.sym not in self.funcs:
            self.declare_func(s)
            
        for stmt in body:
          self.lower_stmt(stmt)

      case syntax.If(test=test, body=body, orelse=orelse):
        cond = self.lower_scalar(test)
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
          self.fb.branch(self.lower_scalar(test), body_b, end_b)

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
          values = tuple(leaves(self.lower_expr(val)))
        else:
          values = ()
        self.fb.set_term(Ret(values))
        self.fb.start(self.fb.new_block("dead"))

      case syntax.Pass():
        pass

      case syntax.TypeDecl() | syntax.StructDecl():
        pass

  def lower_scalar(self, node: syntax.Expression) -> Operand:
    val = self.lower_expr(node)
    assert not isinstance(val, Agg)
    return val

  def lower_expr(self, node: syntax.Expression) -> Val:
    match node:
      case syntax.BinaryExpr(lhs=lhs, rhs=rhs, bop=bop_node):
        dst_v = self.fb.temp(ScalarType.INT)
        lhs_v = self.lower_scalar(lhs)
        rhs_v = self.lower_scalar(rhs)
        self.fb.emit(BinOp(dst_v, lhs_v, binaryop_of(bop_node.bop), rhs_v))

        return dst_v

      case syntax.UnaryExpr(val=val, uop=uop_node):
        v = self.lower_scalar(val)
        dst = self.fb.temp(ScalarType.INT)
        zero = IntImm(0)

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

        call_args: list[Operand] = []
        for a in args:
          call_args.extend(leaves(self.lower_expr(a)))

        rets = [self.fb.temp(t) for t in callee.ret_types]
        self.fb.emit(Call(FuncImm(callee), tuple(call_args), tuple(rets)))

        it = iter(rets)
        return build(layout_of(node.type), lambda _, __: next(it))

      case syntax.Name() | syntax.MemberAccess():
        return self.lower_lval(node)

      case syntax.ScalarLiteral(val=v, type=type):
        layout = layout_of(type)
        assert isinstance(layout, Scalar)

        match layout.type:
          case ScalarType.INT:
            assert isinstance(v, (utils.IntImm, utils.BoolImm))
            return IntImm(int(v.val))

          case ScalarType.STR:
            assert isinstance(v, utils.StrImm)
            return StrImm(v.val)

          case ScalarType.FUNC | ScalarType.LABEL:
            raise NotImplementedError

      case syntax.TupleLiteral(elems=elems):
        return Agg(tuple(
          (str(i), self.lower_expr(e)) for i, e in enumerate(elems)
        ))

      case syntax.StructLiteral(fields=fields):
        vals = {f.name: self.lower_expr(f.val) for f in fields}
        layout = layout_of(node.type)
        assert isinstance(layout, Aggregate)
        return Agg(tuple((name, vals[name]) for name, _ in layout.fields))

  def lower_lval(self, node: syntax.Name | syntax.MemberAccess) -> Val:
    match node:
      case syntax.Name(sym=sym):
        return self.sym_vals[sym]

      case syntax.MemberAccess(src=src, att=att):
        src_v = self.lower_expr(src)
        assert isinstance(src_v, Agg)
        return src_v.get(att)
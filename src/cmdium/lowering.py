from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto

from cmdium import syntax
from cmdium.utils import DataEnum


class Type:
  ...


@dataclass(frozen=True)
class Scalar(Type):
  name: str


@dataclass
class Symbol:
  name: str
  type: Type
  slot: Slot


class BuiltinTypes:
  i32 = Scalar("i32")


def binop_type(op: syntax.BinaryOp, lhs: Type, rhs: Type) -> Type:
  if lhs == rhs == BuiltinTypes.i32:
    return BuiltinTypes.i32

  raise SyntaxError(f"invalid types for binop: {lhs!r} and {rhs!r}")


@dataclass
class Scope:
  parent: Scope | None = None
  types: dict[str, Type] = field(default_factory=dict)
  names: dict[str, Symbol] = field(default_factory=dict)

  def lookup_type(self, name: str) -> Type | None:
    s = self
    while s is not None:
      if name in s.types:
        return s.types[name]
      s = s.parent

    return None

  def lookup_name(self, name: str) -> Symbol | None:
    s = self
    while s is not None:
      if name in s.names:
        return s.names[name]
      s = s.parent

    return None

  def declare_type(self, name: str, type: Type):
    if name in self.types:
      raise SyntaxError(f"type {name!r} already declared")
    self.types[name] = type

  def declare_name(self, name: str, op: Symbol):
    if name in self.names:
      raise SyntaxError(f"name {name!r} already declared")
    self.names[name] = op


def root_scope() -> Scope:
  scope = Scope()
  scope.declare_type("i32", BuiltinTypes.i32)
  return scope


class IR:
  ...


@dataclass
class Copy(IR):
  dst: Slot
  src: Operand


class Operand:
  ...


class BaseKind(DataEnum):
  I32 = auto()


def kind_of(type: Type) -> BaseKind:
  if type == BuiltinTypes.i32:
    return BaseKind.I32

  raise RuntimeError(f"no storage kind for {type!r}")


@dataclass
class Slot(Operand):
  path: str
  kind: BaseKind
  temp: bool = False


@dataclass
class ImmValue(Operand):
  val: int
  kind: BaseKind


@dataclass
class BinOp(IR):
  dst: Slot
  bop: syntax.BinaryOp
  src: Operand


class IRBuilder:
  def __init__(self):
    self.insts: list[IR] = []
    self.n_tmp = -1

  def emit(self, inst: IR):
    self.insts.append(inst)

  def temp_slot(self, kind: BaseKind):
    self.n_tmp += 1
    return Slot(f"t{self.n_tmp}", kind, temp=True)


class Lowerer:
  def __init__(self, tree: syntax.CompilationUnit):
    self.tree = tree
    self.builder = IRBuilder()
    self.scope = root_scope()

  def visit(self) -> list[IR]:
    for stmt in self.tree.body:
      self.visit_stmt(stmt)

    return self.builder.insts

  def resolve_type(self, node: syntax.Expression) -> Type:
    match node:
      case syntax.BinaryExpr():
        return binop_type(
          node.bop,
          self.resolve_type(node.lhs),
          self.resolve_type(node.rhs),
        )

      case syntax.Name():
        symbol = self.scope.lookup_name(node.name)
        if symbol is None:
          raise SyntaxError(f"unknown name: {node.name!r}")
        
        return symbol.type

      case syntax.ScalarLiteral():
        return BuiltinTypes.i32

      case _:
        assert False

  def visit_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl():
        type = self.visit_type(node.type)
        if node.init is not None:
          init, init_type = self.visit_expr(node.init)
          if init_type != type:
            raise RuntimeError(f"cannot init {type!r} with {init_type!r}")
        else:
          init = ImmValue(0, kind_of(type))
        
        slot = Slot(node.name, kind_of(type))
        self.scope.declare_name(node.name, Symbol(node.name, type, slot))
        self.builder.emit(Copy(slot, init))

      case syntax.Assign():
        dst, dst_type = self.visit_lval(node.dst)
        val, val_type = self.visit_expr(node.val)

        if dst_type != val_type:
          raise SyntaxError(
            f"invalid types for assign: {dst_type!r} and {val_type!r}"
          )

        self.builder.emit(Copy(dst, val))

  def visit_expr(self, node: syntax.Expression) -> tuple[Operand, Type]:
    match node:
      case syntax.BinaryExpr():
        lhs, lhs_type = self.visit_expr(node.lhs)
        dst = self.builder.temp_slot(kind_of(lhs_type))
        self.builder.emit(Copy(dst, lhs))

        rhs, rhs_type = self.visit_expr(node.rhs)
        dst_type = binop_type(node.bop, lhs_type, rhs_type)
        self.builder.emit(BinOp(dst, node.bop, rhs))

        return dst, dst_type

      case syntax.Name():
        return self.visit_lval(node)

      case syntax.ScalarLiteral():
        val = int(node.text)
        if not -2147483648 <= val <= 2147483647:
          raise SyntaxError(f"invalid integer size: {val}")

        type = BuiltinTypes.i32
        return ImmValue(val, kind_of(type)), type

      case _:
        assert False

  def visit_lval(self, node: syntax.LeftValue) -> tuple[Slot, Type]:
    match node:
      case syntax.Name():
        sym = self.scope.lookup_name(node.name)
        if sym is None:
          raise SyntaxError(f"unknown name: {node.name!r}")

        return sym.slot, sym.type

      case _:
        assert False

  def visit_type(self, node: syntax.Type) -> Type:
    match node:
      case syntax.NamedType():
        type = self.scope.lookup_type(node.name)
        if type is None:
          raise SyntaxError(f"unknown type: {node.name!r}")

        return type

      case _:
        assert False


class DatapackRenderer:
  def __init__(self, insts: list[IR], ns: str):
    self.insts = insts
    self.ns = ns

    self.consts: set[int] = set()
    self.funcs: dict[str, list[str]] = {}

  def emit(self, cmd: str, func: str):
    self.funcs.setdefault(func, []).append(cmd)

  def holder(self, slot: Slot | int) -> str:
    if isinstance(slot, int):
      return f"%{slot}"

    if slot.temp:
      return f"#{slot.path}"

    return f"${slot.path}"  # avoid conflict with real players

  def render(self) -> dict[str, list[str]]:
    self.emit(f"scoreboard objectives add {self.ns} dummy", func="setup")

    for inst in self.insts:
      self.render_inst(inst)

    self.write_consts()
    return self.funcs

  def add_const(self, const: int):
    self.consts.add(const)

  def write_consts(self):
    for const in sorted(self.consts):
      self.emit(
        f"scoreboard players set {self.holder(const)} {self.ns} {const}",
        func="setup"
      )

  def render_inst(self, inst: IR):
    match inst:
      case Copy():
        self.render_copy(inst)

      case BinOp():
        self.render_binop(inst)

      case _:
        assert False

  def render_copy(self, inst: Copy):
    match inst.dst, inst.src:
      case (
        Slot(kind=BaseKind.I32) as dst,
        Slot(kind=BaseKind.I32) as src,
      ):
        self.emit(
          f"scoreboard players operation "
          f"{self.holder(dst)} {self.ns} = {self.holder(src)} {self.ns}",
          func="main"
        )

      case (
        Slot(kind=BaseKind.I32) as dst,
        ImmValue(kind=BaseKind.I32) as src,
      ):
        self.emit(
          f"scoreboard players set {self.holder(dst)} {self.ns} {src.val}",
          func="main"
        )

      case _:
        assert False

  def render_binop(self, inst: BinOp):
    match inst.dst, inst.bop, inst.src:
      case (
        Slot(kind=BaseKind.I32) as dst,
        syntax.BinaryOp.ADD,
        ImmValue(kind=BaseKind.I32) as src,
      ):
        if src.val >= 0:
          op = "add"
          val = src.val
        else:
          op = "remove"
          val = -src.val
        
        self.emit(
          f"scoreboard players {op} {self.holder(dst)} {self.ns} {val}",
          func="main"
        )

      case (
        Slot(kind=BaseKind.I32) as dst,
        syntax.BinaryOp.SUB,
        ImmValue(kind=BaseKind.I32) as src,
      ):
        if src.val >= 0:
          op = "remove"
          val = src.val
        else:
          op = "add"
          val = -src.val

        self.emit(
          f"scoreboard players {op} {self.holder(dst)} {self.ns} {val}",
          func="main"
        )

      case (
        Slot(kind=BaseKind.I32) as dst,
        bop,
        Slot(kind=BaseKind.I32) | ImmValue(kind=BaseKind.I32) as src,
      ):
        op = {
          syntax.BinaryOp.ADD: "+=",
          syntax.BinaryOp.SUB: "-=",
          syntax.BinaryOp.MUL: "*=",
          syntax.BinaryOp.DIV: "/=",
          syntax.BinaryOp.MOD: "%=",
        }[bop]

        if isinstance(src, ImmValue):
          self.add_const(src.val)
          src_holder = self.holder(src.val)
        else:
          src_holder = self.holder(src)

        self.emit(
          f"scoreboard players operation "
          f"{self.holder(dst)} {self.ns} {op} {src_holder} {self.ns}",
          func="main"
        )

      case _:
        assert False

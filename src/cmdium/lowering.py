from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import auto

from cmdium import syntax
from cmdium.utils import DataEnum


class Type:
  ...


@dataclass(frozen=True)
class Scalar(Type):
  name: str


@dataclass(eq=False)  # nominal type
class Struct(Type):
  name: str
  fields: dict[str, Type]


@dataclass
class Symbol:
  name: str
  place: Place

  @property
  def type(self) -> Type:
    return self.place.type


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
  dst: Place
  src: Operand


class Operand:
  ...


@dataclass(frozen=True)
class Place(Operand):
  path: str
  type: Type
  temp: bool = False

  def project(self, name: str) -> Place:
    if (
      not isinstance(self.type, Struct)
      or name not in self.type.fields
    ):
      raise RuntimeError(f"invalid field name: {name!r}")

    return Place(f"{self.path}.{name}", self.type.fields[name], self.temp)

  def leaves(self) -> Iterator[Place]:
    match self.type:
      case Struct(fields=fields):
        for f in fields:
          yield from self.project(f).leaves()
      
      case Scalar():
        yield self
      
      case _:
        raise RuntimeError(f"unhandled branch: {self.type!r}")


@dataclass
class ImmValue(Operand):
  val: int
  type: Type


@dataclass
class BinOp(IR):
  dst: Place
  bop: syntax.BinaryOp
  src: Operand


class IRBuilder:
  def __init__(self):
    self.insts: list[IR] = []
    self.n_tmp = -1

  def emit(self, inst: IR):
    self.insts.append(inst)

  def alloc(self, name: str, type: Type) -> Place:
    return Place(name, type)

  def temp_place(self, type: Type) -> Place:
    self.n_tmp += 1
    return Place(f"t{self.n_tmp}", type, temp=True)


type Value = LValue | RValue


@dataclass
class LValue:
  val: Place
  type: Type


@dataclass
class RValue:
  val: Operand
  type: Type


class Lowerer:
  def __init__(self, tree: syntax.CompilationUnit):
    self.tree = tree
    self.builder = IRBuilder()
    self.scope = root_scope()

  def visit(self) -> list[IR]:
    for stmt in self.tree.body:
      self.visit_stmt(stmt)

    return self.builder.insts

  def visit_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl():
        type = self.visit_type(node.type)
        init = None
        if node.init is not None:
          init = self.visit_expr(node.init)
          if init.type != type:
            raise RuntimeError(f"cannot init {type!r} with {init.type!r}")

        place = self.builder.alloc(node.name, type)
        self.scope.declare_name(node.name, Symbol(node.name, place))

        if init is not None:
          self.builder.emit(Copy(place, init.val))
        else:
          for leaf in place.leaves():
            self.builder.emit(Copy(leaf, ImmValue(0, leaf.type)))

      case syntax.Assign():
        dst = self.visit_expr(node.dst)
        val = self.visit_expr(node.val)

        if isinstance(dst, RValue):
          raise SyntaxError(f"cannot assign to {node.dst}")
        
        if dst.type != val.type:
          raise SyntaxError(
            f"invalid types for assign: {dst.type!r} and {val.type!r}"
          )

        self.builder.emit(Copy(dst.val, val.val))

      case syntax.StructDecl():
        fields: dict[str, Type] = {}
        for field in node.fields:
          if field.name in fields:
            raise SyntaxError(f"duplicate field {field.name!r}")

          fields[field.name] = self.visit_type(field.type)

        self.scope.declare_type(node.name, Struct(node.name, fields))

      case _:
        raise RuntimeError(f"unhandled statement: {node!r}")

  def visit_expr(self, node: syntax.Expression) -> Value:
    match node:
      case syntax.BinaryExpr():
        lhs = self.visit_expr(node.lhs)
        dst = self.builder.temp_place(lhs.type)
        self.builder.emit(Copy(dst, lhs.val))

        rhs = self.visit_expr(node.rhs)
        dst_type = binop_type(node.bop, lhs.type, rhs.type)
        self.builder.emit(BinOp(dst, node.bop, rhs.val))

        return RValue(dst, dst_type)

      case syntax.Name():
        sym = self.scope.lookup_name(node.name)
        if sym is None:
          raise SyntaxError(f"unknown name: {node.name!r}")

        return LValue(sym.place, sym.type)

      case syntax.MemberAccess():
        src = self.visit_expr(node.src)
        if not (
          isinstance(src.type, Struct)
          and isinstance(src.val, Place)
        ):
          raise SyntaxError(f"{src.type!r} has no fields")

        mem = src.val.project(node.att)
        return src.__class__(mem, mem.type)

      case syntax.ScalarLiteral():
        val = int(node.text)
        if not -2147483648 <= val <= 2147483647:
          raise SyntaxError(f"invalid integer size: {val}")

        type = BuiltinTypes.i32
        imm = ImmValue(val, type)
        return RValue(imm, type)

      case _:
        raise RuntimeError(f"unhandled expression: {node!r}")

  def visit_type(self, node: syntax.Type) -> Type:
    match node:
      case syntax.NamedType():
        type = self.scope.lookup_type(node.name)
        if type is None:
          raise SyntaxError(f"unknown type: {node.name!r}")

        return type

      case _:
        raise RuntimeError(f"unhandled type: {node!r}")


class Slot:
  ...


@dataclass
class PathSlot(Slot):
  path: str
  kind: BaseKind
  temp: bool = False


@dataclass
class ImmSlot(Slot):
  val: int
  kind: BaseKind


def get_pathslot(place: Place):
  return PathSlot(place.path, kind_of(place.type), place.temp)

def get_immslot(imm: ImmValue):
  return ImmSlot(imm.val, kind_of(imm.type))


class BaseKind(DataEnum):
  INT = auto()


def kind_of(type: Type) -> BaseKind:
  if type == BuiltinTypes.i32:
    return BaseKind.INT

  raise RuntimeError(f"no storage kind for {type!r}")


class DatapackRenderer:
  def __init__(self, insts: list[IR], ns: str):
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

  def holder(self, slot: Slot) -> str:
    if isinstance(slot, ImmSlot):
      return self.const_holder(slot.val)

    assert isinstance(slot, PathSlot)
    
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
        f"scoreboard players set {self.const_holder(const)} {self.ns} {const}",
        func="setup"
      )

  def render_inst(self, inst: IR):
    match inst:
      case Copy():
        self.render_copy(inst)

      case BinOp():
        self.render_binop(inst)

      case _:
        raise RuntimeError(f"unhandled instruction: {inst!r}")

  def render_copy(self, inst: Copy):
    match inst.src:
      case ImmValue():
        (dst,) = inst.dst.leaves()
        self.render_copy_scalar(get_pathslot(dst), get_immslot(inst.src))

      case Place():
        for dst, src in zip(inst.dst.leaves(), inst.src.leaves()):
          self.render_copy_scalar(get_pathslot(dst), get_pathslot(src))
          
      case _:
        raise RuntimeError(f"unhandled operand: {inst.src!r}")

  def render_copy_scalar(self, dst: PathSlot, src: Slot):
    match dst, src:
      case (
        PathSlot(kind=BaseKind.INT),
        PathSlot(kind=BaseKind.INT),
      ):
        self.emit(
          f"scoreboard players operation "
          f"{self.holder(dst)} {self.ns} = {self.holder(src)} {self.ns}",
          func="main"
        )

      case (
        PathSlot(kind=BaseKind.INT),
        ImmSlot(kind=BaseKind.INT),
      ):
        self.emit(
          f"scoreboard players set {self.holder(dst)} {self.ns} {src.val}",
          func="main"
        )

      case _:
        raise RuntimeError(f"unhandled slots: {dst!r} and {src!r}")

  def render_binop(self, inst: BinOp):
    match inst.dst, inst.bop, inst.src:
      case (
        Place(type=BuiltinTypes.i32) as dst,
        syntax.BinaryOp.ADD | syntax.BinaryOp.SUB as bop,
        ImmValue(type=BuiltinTypes.i32) as src,
      ):
        op = "add" if (bop == syntax.BinaryOp.ADD) == (src.val >= 0) else "remove"
        val = abs(src.val)

        dst_slot = get_pathslot(dst)
        
        self.emit(
          f"scoreboard players {op} {self.holder(dst_slot)} {self.ns} {val}",
          func="main"
        )

      case (
        Place(type=BuiltinTypes.i32) as dst,
        bop,
        Place(type=BuiltinTypes.i32) | ImmValue(type=BuiltinTypes.i32) as src,
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

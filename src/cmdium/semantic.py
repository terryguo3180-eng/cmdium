from __future__ import annotations

from dataclasses import dataclass, field

from cmdium import error
from cmdium import lowering
from cmdium import syntax


class TypeInfo:
  ...


@dataclass(frozen=True)
class Scalar(TypeInfo):
  name: str


@dataclass(eq=False)  # nominal type
class Struct(TypeInfo):
  name: str
  fields: dict[str, TypeInfo]


@dataclass
class Symbol:
  name: str
  type: TypeInfo

  place: lowering.Place = field(init=False)

  def set_place(self, place: lowering.Place):
    self.place = place


class BuiltinTypes:
  i32 = Scalar("i32")


def binop_type(
  op: syntax.BinaryOpNode, lhs: TypeInfo, rhs: TypeInfo
) -> TypeInfo:
  if lhs == rhs == BuiltinTypes.i32:
    return BuiltinTypes.i32

  raise error.CompileError(
    f"invalid types for this operation: {lhs!r} and {rhs!r}", op.span
  )


@dataclass
class Scope:
  parent: Scope | None = None
  types: dict[str, TypeInfo] = field(default_factory=dict)
  names: dict[str, Symbol] = field(default_factory=dict)

  def lookup_type(self, name: str) -> TypeInfo | None:
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

  def declare_type(self, name: str, type: TypeInfo, span: error.Span):
    if name in self.types:
      raise error.CompileError(f"type {name!r} already declared", span)
    self.types[name] = type

  def declare_name(self, name: str, op: Symbol, span: error.Span):
    if name in self.names:
      raise error.CompileError(f"name {name!r} already declared", span)
    self.names[name] = op


def root_scope(file: str) -> Scope:
  scope = Scope()
  span = error.Span(file, 0, 0)
  scope.declare_type("i32", BuiltinTypes.i32, span)
  return scope


class Semantic:
  def __init__(self, tree: syntax.CompilationUnit, file: str):
    self.tree = tree
    self.scope = root_scope(file)

  def analyze(self) -> syntax.CompilationUnit:
    for stmt in self.tree.body:
      self.analyze_stmt(stmt)

    return self.tree

  def analyze_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl():
        type = self.analyze_type(node.type)
        if node.init is not None:
          self.analyze_expr(node.init)
          init_ty = node.init.type
          if init_ty != type:
            raise error.CompileError(
              f"cannot init declared type {type!r} with {init_ty!r}",
              node.init.span,
            )

        sym = Symbol(node.name, type)
        node.set_symbol(sym)
        self.scope.declare_name(node.name, sym, node.span)

      case syntax.Assign():
        self.analyze_expr(node.dst)
        self.analyze_expr(node.val)

        if not node.dst.lval:
          raise error.CompileError(
            f"cannot assign to this expression", node.dst.span
          )

        dst_ty, val_ty = node.dst.type, node.val.type
        
        if dst_ty != val_ty:
          raise error.CompileError(
            f"invalid types for assignment: {dst_ty!r} and {val_ty!r}",
            node.val.span,
          )

      case syntax.TypeDecl():
        self.scope.declare_type(
          node.name, self.analyze_type(node.type), node.span
        )
      
      case syntax.StructDecl():
        fields: dict[str, TypeInfo] = {}
        for field in node.fields:
          if field.name in fields:
            raise error.CompileError(
              f"duplicate field {field.name!r}", field.span,
            )

          fields[field.name] = self.analyze_type(field.type)

        self.scope.declare_type(node.name, Struct(node.name, fields), node.span)

      case _:
        raise RuntimeError(f"unhandled statement: {node!r}")


  def analyze_expr(self, node: syntax.Expression):
    match node:
      case syntax.BinaryExpr():
        self.analyze_expr(node.lhs)
        self.analyze_expr(node.rhs)
        dst_ty = binop_type(node.bop, node.lhs.type, node.rhs.type)
        node.set_typeinfo(dst_ty, False)

      case syntax.Name():
        sym = self.scope.lookup_name(node.name)
        if sym is None:
          raise error.CompileError(f"undeclared name {node.name!r}", node.span)

        node.set_symbol(sym)
        node.set_typeinfo(sym.type, True)

      case syntax.MemberAccess():
        src, att = node.src, node.att

        self.analyze_expr(src)
        if not isinstance(src.type, Struct):
          raise error.CompileError(f"{src.type!r} has no fields", node.span)
        
        if att not in src.type.fields:
          raise error.CompileError(
            f"undeclared field name: {node.att!r}", node.span
          )

        node.set_typeinfo(src.type.fields[att], src.lval)

      case syntax.ScalarLiteral():
        val = int(node.text)
        if not -2147483648 <= val <= 2147483647:
          raise error.CompileError(f"invalid integer size: {val}", node.span)

        node.set_typeinfo(BuiltinTypes.i32, False)

      case _:
        raise RuntimeError(f"unhandled expression: {node!r}")

  def analyze_type(self, node: syntax.TypeExpr) -> TypeInfo:    
    match node:
      case syntax.NamedType():
        type = self.scope.lookup_type(node.name)
        if type is None:
          raise error.CompileError(f"undeclared type: {node.name!r}", node.span)

        return type

      case _:
        raise RuntimeError(f"unhandled type: {node!r}")

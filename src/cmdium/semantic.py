from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto
from typing import Any

from cmdium import error
from cmdium import syntax
from cmdium import utils


class PrimKind(utils.DataEnum):
  I32 = auto()
  BOOL = auto()
  STR = auto()


@dataclass(frozen=True)
class Prim:
  kind: PrimKind

  def __str__(self):
    return self.kind.name.lower()


@dataclass(eq=False)  # nominal type
class Struct:
  name: str
  fields: dict[str, TypeInfo]

  def __str__(self):
    return self.name


@dataclass(eq=True)  # structual type
class Tuple:
  fields: tuple[TypeInfo, ...]

  def __str__(self):
    return f"({', '.join(str(f) for f in self.fields)})"


@dataclass
class Function:
  params: list[TypeInfo]
  ret_type: TypeInfo

  def __str__(self):
    return f"({", ".join(str(p) for p in self.params)}) -> {self.ret_type!s}"


TypeInfo = Prim | Struct | Tuple | Function


@dataclass(eq=False)
class Symbol:
  name: str
  type: TypeInfo


class BuiltinTypes:
  i32 = Prim(PrimKind.I32)
  bool = Prim(PrimKind.BOOL)
  void = Tuple(())
  str = Prim(PrimKind.STR)


def binop_type(
  op: syntax.BinaryOpNode, lhs: TypeInfo, rhs: TypeInfo
) -> TypeInfo:
  match op.bop, lhs, rhs:
    case (
      syntax.BinaryOp.ADD | syntax.BinaryOp.SUB |
      syntax.BinaryOp.MUL | syntax.BinaryOp.DIV |
      syntax.BinaryOp.MOD
    ), Prim(PrimKind.I32), Prim(PrimKind.I32):
      return BuiltinTypes.i32

    case (
      syntax.BinaryOp.CGT | syntax.BinaryOp.CLT |
      syntax.BinaryOp.CGE | syntax.BinaryOp.CLE
    ), Prim(PrimKind.I32), Prim(PrimKind.I32):
      return BuiltinTypes.bool

    case (
      syntax.BinaryOp.CEQ | syntax.BinaryOp.CNE
    ), Prim(PrimKind.I32), Prim(PrimKind.I32):
      return BuiltinTypes.bool

    case (
      syntax.BinaryOp.CEQ | syntax.BinaryOp.CNE
    ), Prim(PrimKind.BOOL), Prim(PrimKind.BOOL):
      return BuiltinTypes.bool

    case (
      syntax.BinaryOp.CEQ | syntax.BinaryOp.CNE
    ), Prim(PrimKind.STR), Prim(PrimKind.STR):
      return BuiltinTypes.bool

    case (
      syntax.BinaryOp.OR | syntax.BinaryOp.AND
    ), Prim(PrimKind.BOOL), Prim(PrimKind.BOOL):
      return BuiltinTypes.bool

    case _:
      raise error.CompileError(
        f"invalid types for this operation: {lhs!s} and {rhs!s}", op.span
      )


def unaop_type(op: syntax.UnaryOpNode, val: TypeInfo) -> TypeInfo:
  match op.uop, val:
    case syntax.UnaryOp.NEG, Prim(PrimKind.I32):
      return BuiltinTypes.i32

    case syntax.UnaryOp.NOT, Prim(PrimKind.BOOL):
      return BuiltinTypes.bool

    case _:
      raise error.CompileError(
        f"invalid type for this operation: {val!s}", op.span
      )


@dataclass
class Scope:
  parent: Scope | None = None
  types: dict[str, TypeInfo] = field(default_factory=dict[str, TypeInfo])
  names: dict[str, Symbol] = field(default_factory=dict[str, Symbol])

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

  for name, type in vars(BuiltinTypes).items():
    if isinstance(type, TypeInfo):
      scope.declare_type(name, type, span)
  
  return scope


class Semantic:
  def __init__(self, tree: syntax.CompilationUnit, file: str):
    self.tree = tree
    self.scope = root_scope(file)

    self.inside_loop = False
    self.inside_func = False
    self.return_type: TypeInfo | None = None

  def analyze(self) -> syntax.CompilationUnit:
    for stmt in self.tree.body:
      self.analyze_stmt(stmt)

    return self.tree

  @contextmanager
  def with_scope(self):
    prev = self.scope
    self.scope = Scope(prev)
    try:
      yield self.scope
    finally:
      self.scope = prev

  @contextmanager
  def with_state(
    self, **kwargs: Any
  ) -> Generator[dict[str, Any], None, None]:
    prevs: dict[str, Any] = {}
    for key, val in kwargs.items():
      prevs[key] = getattr(self, key)
      setattr(self, key, val)
    try:
      yield prevs
    finally:
      for key, val in prevs.items():
        setattr(self, key, val)

  def analyze_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl(
        name=name, type=type, init=init, name_tok=name_tok, span=span
      ):
        ty = self.analyze_type(type)
        if init is not None:
          self.analyze_expr(init)
          if init.type != ty:
            raise error.CompileError(
              f"type {init.type!s} is not assignable to declared type {ty!s}",
              init.span
            )

        sym = Symbol(name, ty)
        node.set_symbol(sym)
        self.scope.declare_name(name, sym, name_tok.span)

      case syntax.Assign(dst=dst, val=val, span=span):
        self.analyze_expr(dst)
        self.analyze_expr(val)

        if not dst.lval:
          raise error.CompileError(
            "cannot assign to this expression", dst.span
          )

        dst_ty, val_ty = dst.type, val.type
        
        if dst_ty != val_ty:
          raise error.CompileError(
            f"type {val_ty!s} is not assignable to declared type {dst_ty!s}",
            span
          )

      case syntax.AugAssign(dst=dst, bop=bop, val=val):
        self.analyze_expr(dst)
        self.analyze_expr(val)

        if not dst.lval:
          raise error.CompileError(
            "cannot assign to this expression", dst.span
          )

        dst_ty, val_ty = dst.type, val.type
        binop_type(bop, dst_ty, val_ty)

      case syntax.Block(body=body):
        with self.with_scope():
          for stmt in body:
            self.analyze_stmt(stmt)

      case syntax.Break(span=span):
        if not self.inside_loop:
          raise error.CompileError("break statement outside loop", span)

      case syntax.Continue(span=span):
        if not self.inside_loop:
          raise error.CompileError("continue statement outside loop", span)

      case syntax.TypeDecl(name_tok=name_tok):
        self.scope.declare_type(
          node.name, self.analyze_type(node.type), name_tok.span
        )
      
      case syntax.StructDecl(
        name=name, fields=fields, name_tok=name_tok,
      ):
        fields_map: dict[str, TypeInfo] = {}
        for field in fields:
          if field.name in fields_map:
            raise error.CompileError(
              f"duplicate field {field.name!r}", field.span
            )

          fields_map[field.name] = self.analyze_type(field.type)

        ty = Struct(name, fields_map)
        self.scope.declare_type(name, ty, name_tok.span)

      case syntax.FuncDecl(
        name=name, params=params, ret_type=ret_type, body=body,
        name_tok=name_tok,
      ):
        if ret_type is None:
          return_type = BuiltinTypes.void
        else:
          return_type = self.analyze_type(ret_type)

        param_types: list[TypeInfo] = []
        for p in params:
          ty = self.analyze_type(p.type)
          param_types.append(ty)

        ty = Function(param_types, return_type)
        sym = Symbol(name, ty)
        node.set_symbol(sym)
        self.scope.declare_name(name, sym, name_tok.span)
        
        with self.with_scope():
          for p, pty in zip(params, param_types):
            psym = Symbol(p.name, pty)
            p.set_symbol(psym)
            self.scope.declare_name(p.name, psym, p.span)

          with self.with_state(return_type=return_type, inside_func=True):
            self.analyze_stmt(body)

      case syntax.ExprStmt(val=val):
        self.analyze_expr(val)

      case syntax.If(test=test, body=body, orelse=orelse):
        self.analyze_expr(test)
        if test.type != BuiltinTypes.bool:
          raise error.CompileError(
            f"invalid type for if condition: {test.type!s} (must be bool)",
            test.span
          )

        self.analyze_stmt(body)
        if orelse is not None:
          self.analyze_stmt(orelse)

      case syntax.Pass():
        pass

      case syntax.Return(val=val, span=span):
        if not self.inside_func:
          raise error.CompileError("return statement outside function", span)

        if val is None:
          val_ty = BuiltinTypes.void
          val_span = node.span
        else:
          self.analyze_expr(val)
          val_ty = val.type
          val_span = val.span

        if val_ty != self.return_type:
          raise error.CompileError(
            f"type {val_ty} is not assignable to declared return type "
            + str(self.return_type), val_span
          )

      case syntax.While(test=test, body=body):
        if test is not None:
          self.analyze_expr(test)
          if test.type != BuiltinTypes.bool:
            raise error.CompileError(
              f"invalid type for while condition: {test.type!s} (must be bool)",
              test.span
            )

        with self.with_state(inside_loop=True):
          self.analyze_stmt(body)

  def analyze_expr(self, node: syntax.Expression):
    match node:
      case syntax.BinaryExpr(lhs=lhs, bop=bop, rhs=rhs):
        self.analyze_expr(lhs)
        self.analyze_expr(rhs)
        dst_ty = binop_type(bop, lhs.type, rhs.type)
        node.set_typeinfo(dst_ty, False)

      case syntax.UnaryExpr(uop=uop, val=val):
        self.analyze_expr(val)
        dst_ty = unaop_type(uop, val.type)
        node.set_typeinfo(dst_ty, False)

      case syntax.Name(name=name, span=span):
        sym = self.scope.lookup_name(name)
        if sym is None:
          raise error.CompileError(f"undeclared name {name!r}", span)

        node.set_symbol(sym)
        node.set_typeinfo(sym.type, True)

      case syntax.MemberAccess(src=src, att=att, span=span):
        self.analyze_expr(src)
        if not isinstance(src.type, Struct):
          raise error.CompileError(f"type {src.type!s} has no fields", span)
        
        if att not in src.type.fields:
          raise error.CompileError(f"undeclared field name: {att!r}", span)

        node.set_typeinfo(src.type.fields[att], src.lval)

      case syntax.Call(src=src, args=args, span=span):
        self.analyze_expr(src)
        if not isinstance(src.type, Function):
          raise error.CompileError(
            f"type {src.type!s} is not callable", src.span
          )

        arg_types: list[TypeInfo] = []
        for arg in args:
          self.analyze_expr(arg)
          arg_types.append(arg.type)

        if arg_types != src.type.params:
          raise error.CompileError(
            f"argument type(s) ({", ".join(str(t) for t in arg_types)}) "
            f"are not assignable to declared type(s) ("
            + ", ".join(str(t) for t in src.type.params) + ")", span
          )

        node.set_typeinfo(src.type.ret_type, False)

      case syntax.ScalarLiteral(val=val, span=span):
        match val:
          case utils.IntImm(val=val):
            if not -2147483648 <= val <= 2147483647:
              raise error.CompileError(f"invalid integer size: {val}", span)

            node.set_typeinfo(BuiltinTypes.i32, False)

          case utils.BoolImm():
            node.set_typeinfo(BuiltinTypes.bool, False)

          case utils.StrImm(val=val):
            node.set_typeinfo(BuiltinTypes.str, False)

  def analyze_type(self, node: syntax.TypeExpr) -> TypeInfo:    
    match node:
      case syntax.NamedType(name=name, span=span):
        type = self.scope.lookup_type(name)
        if type is None:
          raise error.CompileError(f"undeclared type: {name!r}", span)

        return type

      case syntax.TupleType(fields=fields, span=span):
        return Tuple(tuple(self.analyze_type(f) for f in fields))

      case syntax.FuncType(params=params, ret_type=ret_type):
        return Function(
          [self.analyze_type(p) for p in params],
          self.analyze_type(ret_type)
        )

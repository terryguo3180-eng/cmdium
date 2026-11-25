from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto
from typing import Any, Self

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


@dataclass(frozen=True)
class Ref:
  """A pointer to a storage cell; only references can be nullable."""

  inner: TypeInfo
  nullable: bool = False

  def __str__(self):
    return f"&{self.inner!s}{'?' if self.nullable else ''}"


@dataclass(frozen=True)
class Null:
  """The type of the `N` literal: assignable to any nullable reference."""

  def __str__(self):
    return "N"


TypeInfo = Prim | Struct | Tuple | Function | Ref | Null


def assignable(src: TypeInfo, dst: TypeInfo) -> bool:
  """`src` may be stored where `dst` is expected.

  A nullable reference accepts a plain reference (and `N`), but not the
  other way round.
  """
  if src == dst:
    return True

  if isinstance(src, Null):
    return isinstance(dst, Ref) and dst.nullable

  if isinstance(src, Ref) and isinstance(dst, Ref):
    return src.inner == dst.inner and (dst.nullable or not src.nullable)

  return False


@dataclass(eq=False)
class Symbol:
  name: str
  type: TypeInfo
  is_func: bool = False  # declared with `func`, not by a variable
  exported: bool = False  # its scoreboard holder is `$name`, not `#name`


class BuiltinTypes:
  i32 = Prim(PrimKind.I32)
  bool = Prim(PrimKind.BOOL)
  void = Tuple(())
  str = Prim(PrimKind.STR)


@dataclass
class AttributeBase:
  span: error.Span = field(init=False, repr=False)

  def write_span(self, span: error.Span) -> Self:
    self.span = span
    return self


@dataclass
class Export(AttributeBase):
  hint: str | None = None

  def __str__(self):
    return "export"


@dataclass
class Dummy(AttributeBase):
  def __str__(self):
    return "dummy"


Attribute = Export | Dummy


def binop_type(
  op: syntax.BinaryOpNode, lhs: TypeInfo, rhs: TypeInfo
) -> TypeInfo:
  if op.bop in (syntax.BinaryOp.CEQ, syntax.BinaryOp.CNE):
    if isinstance(lhs, Ref) and isinstance(rhs, Ref):
      if lhs.inner != rhs.inner:
        raise error.CompileError(
          f"cannot compare references to {lhs.inner!s} and {rhs.inner!s}",
          op.span,
        )
      return BuiltinTypes.bool

    if isinstance(lhs, Null) or isinstance(rhs, Null):
      other = rhs if isinstance(lhs, Null) else lhs
      if not isinstance(other, Ref):
        raise error.CompileError(
          f"cannot compare {other!s} with N", op.span
        )
      return BuiltinTypes.bool

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


def literal_int(node: syntax.Expression) -> int | None:
  """Value of an integer literal, including a negated one; None otherwise."""
  match node:
    case syntax.ScalarLiteral(val=utils.IntImm(val=val)):
      return val

    case syntax.UnaryExpr(
      uop=syntax.UnaryOpNode(uop=syntax.UnaryOp.NEG), val=inner
    ):
      val = literal_int(inner)
      return None if val is None else -val

    case _:
      return None


def check_divisor(op: syntax.BinaryOpNode, rhs: syntax.Expression) -> None:
  """Reject a constant zero divisor.

  `/=` and `%=` fail at runtime when the divisor is 0 (verified on 1.21.1),
  which would silently leave the destination unchanged.
  """
  if op.bop not in (syntax.BinaryOp.DIV, syntax.BinaryOp.MOD):
    return

  if literal_int(rhs) != 0:
    return

  what = "division" if op.bop is syntax.BinaryOp.DIV else "modulo"
  raise error.CompileError(f"{what} by zero", rhs.span)


@dataclass
class Scope:
  parent: Scope | None = None
  types: dict[str, TypeInfo] = field(default_factory=dict[str, TypeInfo])
  names: dict[str, Symbol] = field(default_factory=dict[str, Symbol])
  narrowed: dict[Symbol, TypeInfo] = field(
    default_factory=dict[Symbol, TypeInfo]
  )

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

  def lookup_narrow(self, sym: Symbol) -> TypeInfo | None:
    """A narrowed type for `sym` (e.g. after `if (p != N)`), if any."""
    s = self
    while s is not None:
      if sym in s.narrowed:
        return s.narrowed[sym]
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
    self.declare_items(self.tree.body)

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
  def with_narrowed(self, narrowed: dict[Symbol, TypeInfo]):
    """Give some symbols a more precise type inside this block."""
    if not narrowed:
      yield self.scope
      return

    prev = self.scope
    self.scope = Scope(prev)
    self.scope.narrowed.update(narrowed)
    try:
      yield self.scope
    finally:
      self.scope = prev

  def narrowing(
    self, test: syntax.Expression
  ) -> tuple[dict[Symbol, TypeInfo], dict[Symbol, TypeInfo]]:
    """Types refined by `p != N` / `p == N` for each branch."""
    none: dict[Symbol, TypeInfo] = {}

    if not isinstance(test, syntax.BinaryExpr):
      return none, none

    op = test.bop.bop
    if op not in (syntax.BinaryOp.CEQ, syntax.BinaryOp.CNE):
      return none, none

    name: syntax.Name | None = None
    if isinstance(test.lhs, syntax.Name) and isinstance(
      test.rhs, syntax.NullLiteral
    ):
      name = test.lhs
    elif isinstance(test.rhs, syntax.Name) and isinstance(
      test.lhs, syntax.NullLiteral
    ):
      name = test.rhs

    if name is None:
      return none, none

    ty = self.scope.lookup_narrow(name.sym) or name.sym.type
    if not (isinstance(ty, Ref) and ty.nullable):
      return none, none

    narrowed: dict[Symbol, TypeInfo] = {name.sym: Ref(ty.inner)}
    return (narrowed, none) if op is syntax.BinaryOp.CNE else (none, narrowed)

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

  def declare_items(self, stmts: list[syntax.Statement]) -> None:
    structs = [s for s in stmts if isinstance(s, syntax.StructDecl)]
    aliases = [s for s in stmts if isinstance(s, syntax.TypeDecl)]
    funcs = [s for s in stmts if isinstance(s, syntax.FuncDecl)]

    shells: list[tuple[syntax.StructDecl, Struct]] = []
    for s in structs:
      ty = Struct(s.name, {})
      self.scope.declare_type(s.name, ty, s.name_tok.span)
      shells.append((s, ty))

    for a in aliases:
      self.scope.declare_type(
        a.name, self.analyze_type(a.type), a.name_tok.span
      )

    for s, ty in shells:
      for f in s.fields:
        if f.name in ty.fields:
          raise error.CompileError(f"duplicate field {f.name!r}", f.span)
        ty.fields[f.name] = self.analyze_type(f.type)

    for s, ty in shells:
      if any(self._contains(ft, ty, set()) for ft in ty.fields.values()):
        raise error.CompileError(
          f"struct {s.name!r} contains itself and would have infinite size",
          s.name_tok.span,
        )

    for fn in funcs:
      if fn.ret_type is None:
        ret = BuiltinTypes.void
      else:
        ret = self.analyze_type(fn.ret_type)
      params = [self.analyze_type(p.type) for p in fn.params]
      sym = Symbol(fn.name, Function(params, ret), is_func=True)
      fn.set_symbol(sym)
      self.scope.declare_name(fn.name, sym, fn.name_tok.span)

  def _contains(self, ty: TypeInfo, target: Struct, seen: set[int]) -> bool:
    match ty:
      case Struct(fields=fields):
        if ty is target:
          return True
        if id(ty) in seen:
          return False
        seen.add(id(ty))
        return any(self._contains(f, target, seen) for f in fields.values())
      
      case Tuple(fields=fields):
        return any(self._contains(f, target, seen) for f in fields)
      
      case _:
        return False

  def check_attrs(
    self,
    node: syntax.Statement,
    allowed: set[type[Attribute]],
    allow_dup: bool = False,
  ):
    attrs = [self.analyze_attr(a) for a in node.parsed_attrs]
    seen: set[type[Attribute]] = set()

    for attr in attrs:
      ta = type(attr)
  
      if ta not in allowed:
        raise error.CompileError(
          f"attribute '{attr!s}' is not supported for this statement",
          attr.span
        )

      if ta in seen and not allow_dup:
        raise error.CompileError(f"duplicate attribute '{attr!s}'", attr.span)

      seen.add(ta)

    node.set_attrs(attrs)

  def analyze_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl(
        name=name, type=type, init=init, name_tok=name_tok, span=span
      ):
        self.check_attrs(node, {Export})

        ty = self.analyze_type(type)
        if init is not None:
          self.analyze_expr(init)
          if not assignable(init.type, ty):
            raise error.CompileError(
              f"type {init.type!s} is not assignable to declared type {ty!s}",
              init.span
            )
        sym = Symbol(name, ty)
        for attr in node.attrs:
          if isinstance(attr, Export):
            sym.exported = True

        node.set_symbol(sym)
        self.scope.declare_name(name, sym, name_tok.span)

      case syntax.Assign(dst=dst, val=val, span=span):
        self.check_attrs(node, set())

        self.analyze_expr(dst)
        self.analyze_expr(val)

        if not dst.lval:
          raise error.CompileError(
            "cannot assign to this expression", dst.span
          )

        if isinstance(dst, syntax.Name) and dst.sym.is_func:
          raise error.CompileError(
            f"cannot assign to function {dst.name!r}", dst.span
          )

        dst_ty, val_ty = dst.type, val.type
        
        if not assignable(val_ty, dst_ty):
          raise error.CompileError(
            f"type {val_ty!s} is not assignable to declared type {dst_ty!s}",
            span
          )

      case syntax.AugAssign(dst=dst, bop=bop, val=val):
        self.check_attrs(node, set())

        self.analyze_expr(dst)
        self.analyze_expr(val)

        if not dst.lval:
          raise error.CompileError(
            "cannot assign to this expression", dst.span
          )

        dst_ty, val_ty = dst.type, val.type
        binop_type(bop, dst_ty, val_ty)

      case syntax.Block(body=body):
        self.check_attrs(node, set())

        with self.with_scope():
          self.declare_items(body)
          for stmt in body:
            self.analyze_stmt(stmt)

      case syntax.Break(span=span):
        self.check_attrs(node, set())

        if not self.inside_loop:
          raise error.CompileError("break statement outside loop", span)

      case syntax.Continue(span=span):
        self.check_attrs(node, set())

        if not self.inside_loop:
          raise error.CompileError("continue statement outside loop", span)

      case syntax.StructDecl() | syntax.TypeDecl():
        self.check_attrs(node, set())
        pass  # declared by declare_items

      case syntax.FuncDecl(params=params, body=body):
        self.check_attrs(node, set())

        fty = node.sym.type  # set by declare_items
        assert isinstance(fty, Function)
        with self.with_scope():
          for p, pty in zip(params, fty.params):
            psym = Symbol(p.name, pty)
            p.set_symbol(psym)
            self.scope.declare_name(p.name, psym, p.span)

          with self.with_state(
            return_type=fty.ret_type, inside_func=True, inside_loop=False
          ):
            self.analyze_stmt(body)

      case syntax.ExprStmt(val=val):
        self.check_attrs(node, set())

        self.analyze_expr(val)

      case syntax.If(test=test, body=body, orelse=orelse):
        self.check_attrs(node, set())

        self.analyze_expr(test)
        if test.type != BuiltinTypes.bool:
          raise error.CompileError(
            f"invalid type for if condition: {test.type!s} (must be bool)",
            test.span
          )

        then_narrow, else_narrow = self.narrowing(test)

        with self.with_narrowed(then_narrow):
          self.analyze_stmt(body)

        if orelse is not None:
          with self.with_narrowed(else_narrow):
            self.analyze_stmt(orelse)

      case syntax.Pass():
        self.check_attrs(node, set())

      case syntax.Return(val=val, span=span):
        self.check_attrs(node, set())

        if not self.inside_func:
          raise error.CompileError("return statement outside function", span)

        if val is None:
          val_ty = BuiltinTypes.void
          val_span = node.span
        else:
          self.analyze_expr(val)
          val_ty = val.type
          val_span = val.span

        ret_type = self.return_type
        assert ret_type is not None

        if not assignable(val_ty, ret_type):
          raise error.CompileError(
            f"type {val_ty} is not assignable to declared return type "
            + str(ret_type), val_span
          )

      case syntax.While(test=test, body=body):
        self.check_attrs(node, set())
        
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
        check_divisor(bop, rhs)
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
        narrowed = self.scope.lookup_narrow(sym)
        node.set_typeinfo(narrowed or sym.type, True)

      case syntax.AddrOf(val=val, span=span):
        self.analyze_expr(val)

        if isinstance(val, syntax.Name) and val.sym.is_func:
          raise error.CompileError(
            "cannot take a reference to a function", span
          )

        if not val.lval:
          raise error.CompileError(
            "cannot take a reference to a value (only lvalues have an address)",
            val.span
          )

        root = val
        while isinstance(root, syntax.MemberAccess):
          root = root.src

        if (
          isinstance(root, syntax.Name)
          and root.sym.exported
        ):
          raise error.CompileError(
            "cannot take a reference to an exported variable: its scoreboard "
            "holder is named '$name', but a pointer only carries the plain "
            "name", span
          )

        node.set_typeinfo(Ref(val.type), True)

      case syntax.Deref(val=val, span=span):
        self.analyze_expr(val)

        match val.type:
          case Ref(inner=inner, nullable=True):
            raise error.CompileError(
              "cannot dereference a nullable reference; compare it with N "
              "first (if (p != N) { ... })", span
            )

          case Ref(inner=inner):
            node.set_typeinfo(inner, True)

          case _ as ty:
            raise error.CompileError(
              f"type {ty!s} is not a reference", span
            )

      case syntax.NullLiteral():
        node.set_typeinfo(Null(), False)

      case syntax.MemberAccess(src=src, att=att, span=span):
        self.analyze_expr(src)

        match src.type:
          case Tuple(fields=fields):
            if not att.isdigit() or int(att) >= len(fields):
              raise error.CompileError(f"tuple has no field {att!r}", span)
            node.set_typeinfo(fields[int(att)], src.lval)

          case Struct(fields=fields):
            if att not in fields:
              raise error.CompileError(f"undeclared field name: {att!r}", span)
            node.set_typeinfo(fields[att], src.lval)

          case Ref():
            raise error.CompileError(
              f"cannot read field {att!r} through a reference; dereference it "
              "first (p&." + att + ")", span
            )

          case Null():
            raise error.CompileError("N has no fields", span)
        
          case _ as ty:
            raise error.CompileError(f"type {ty!s} has no fields", span)

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

        if (
          len(arg_types) != len(src.type.params)
          or not all(
            assignable(a, p) for a, p in zip(arg_types, src.type.params)
          )
        ):
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

      case syntax.TupleLiteral(elems=elems):
        for e in elems:
          self.analyze_expr(e)
        node.set_typeinfo(Tuple(tuple(e.type for e in elems)), False)

      case syntax.StructLiteral(name=name, fields=fields, span=span):
        ty = self.scope.lookup_type(name)
        if not isinstance(ty, Struct):
          raise error.CompileError(f"{name!r} is not a struct type", span)

        seen: set[str] = set()
        for f in fields:
          self.analyze_expr(f.val)
          if f.name not in ty.fields:
            raise error.CompileError(
              f"undeclared field name: {f.name!r}", f.span
            )

          if f.name in seen:
            raise error.CompileError(f"duplicate field {f.name!r}", f.span)

          if not assignable(f.val.type, ty.fields[f.name]):
            raise error.CompileError(
              f"type {f.val.type!s} is not assignable to field type "
              f"{ty.fields[f.name]!s}", f.val.span
            )

          seen.add(f.name)

        if missing := ty.fields.keys() - seen:
          raise error.CompileError(
            f"missing field(s): {sorted(missing)}", span
          )

        node.set_typeinfo(ty, False)

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

      case syntax.RefType(inner=inner):
        return Ref(self.analyze_type(inner))

      case syntax.NullableType(inner=inner, span=span):
        inner_ty = self.analyze_type(inner)
        if not isinstance(inner_ty, Ref):
          raise error.CompileError(
            "only references can be nullable", span
          )
        return Ref(inner_ty.inner, nullable=True)

  def analyze_attr(self, node: syntax.ParsedAttr) -> Attribute:
    match node.name:
      case "export":
        match node.args:
          case []:
            return Export().write_span(node.span)

          case [syntax.ScalarLiteral(val=utils.StrImm(val=name))]:
            return Export(name).write_span(node.span)

          case _:
            raise error.CompileError(
              f"invalid args for attribute {node.name!r}", node.span
            )

      case "dummy":
        return Dummy().write_span(node.span)

      case _:
        raise error.CompileError(
          f"unknown attribute: {node.name!r}", node.name_tok.span
        )
      

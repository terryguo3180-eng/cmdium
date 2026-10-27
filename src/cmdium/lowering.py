from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cmdium import semantic
from cmdium import syntax


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
  type: semantic.TypeInfo
  temp: bool = False

  def project(self, name: str) -> Place:
    assert isinstance(self.type, semantic.Struct)
    return Place(f"{self.path}.{name}", self.type.fields[name], self.temp)

  def leaves(self) -> Iterator[Place]:
    match self.type:
      case semantic.Struct(fields=fields):
        for f in fields:
          yield from self.project(f).leaves()
      
      case semantic.Scalar():
        yield self
      
      case _:
        raise RuntimeError(f"unhandled branch: {self.type!r}")


@dataclass
class ImmValue(Operand):
  val: int
  type: semantic.TypeInfo


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

  def alloc(self, name: str, type: semantic.TypeInfo) -> Place:
    return Place(name, type)

  def temp_place(self, type: semantic.TypeInfo) -> Place:
    self.n_tmp += 1
    return Place(f"t{self.n_tmp}", type, temp=True)


class Lowering:
  def __init__(self, tree: syntax.CompilationUnit):
    self.tree = tree
    self.builder = IRBuilder()

  def lower(self) -> list[IR]:
    for stmt in self.tree.body:
      self.lower_stmt(stmt)

    return self.builder.insts

  def lower_stmt(self, node: syntax.Statement):
    match node:
      case syntax.VarDecl():
        if node.init is not None:
          init = self.lower_expr(node.init)
        else:
          init = None

        sym = node.sym
        place = self.builder.alloc(node.name, sym.type)
        sym.set_place(place)

        if init is not None:
          self.builder.emit(Copy(place, init))
        else:
          for leaf in place.leaves():
            self.builder.emit(Copy(leaf, ImmValue(0, leaf.type)))

      case syntax.Assign():
        dst = self.lower_expr(node.dst)
        val = self.lower_expr(node.val)

        assert isinstance(dst, Place)
        self.builder.emit(Copy(dst, val))

      case syntax.TypeDecl() | syntax.StructDecl():
        pass

      case _:
        raise RuntimeError(f"unhandled statement: {node!r}")

  def lower_expr(self, node: syntax.Expression) -> Operand:
    match node:
      case syntax.BinaryExpr():
        lhs = self.lower_expr(node.lhs)
        dst = self.builder.temp_place(node.lhs.type)
        self.builder.emit(Copy(dst, lhs))

        rhs = self.lower_expr(node.rhs)
        self.builder.emit(BinOp(dst, node.bop.bop, rhs))

        return dst

      case syntax.Name():
        return node.sym.place

      case syntax.MemberAccess():
        src = self.lower_expr(node.src)
        assert isinstance(src, Place)
        return src.project(node.att)

      case syntax.ScalarLiteral():
        val = int(node.text)
        imm = ImmValue(val, node.type)
        return imm

      case _:
        raise RuntimeError(f"unhandled expression: {node!r}")

from __future__ import annotations

from dataclasses import dataclass

from cmdium import syntax


@dataclass
class IR:
  ...


@dataclass
class Copy(IR):
  dst: Place
  src: Operand


@dataclass
class Operand:
  ...


@dataclass
class Place(Operand):
  ...


@dataclass
class ScoreSlot(Place):
  holder: str


@dataclass
class ImmValue(Operand):
  val: int


@dataclass
class BinOp(IR):
  dst: Place
  bop: syntax.BinaryOp
  src: Operand


class IRBuilder:
  def __init__(self):
    self.insts: list[IR] = []
    self.n_tmp = -1
    self.names: dict[str, Operand] = {}

  def emit(self, inst: IR):
    self.insts.append(inst)

  def new_tmp(self):
    self.n_tmp += 1
    return ScoreSlot(f"#t{self.n_tmp}")

  def add_name(self, name: str):
    self.names[name] = ScoreSlot(name)

  def get_name(self, name: str) -> Operand | None:
    return self.names.get(name)


class Lowerer:
  def __init__(self, tree: syntax.CompilationUnit):
    self.tree = tree
    self.builder = IRBuilder()

  def visit(self) -> list[IR]:
    for stmt in self.tree.body:
      self.visit_stmt(stmt)

    return self.builder.insts

  def visit_stmt(self, node: syntax.Statement):
    match node:
      case syntax.Assign():
        dst = ScoreSlot(node.dst)
        self.builder.add_name(node.dst)
        val = self.visit_expr(node.val)
        self.builder.emit(Copy(dst, val))

  def visit_expr(self, node: syntax.Expression) -> Operand:
    match node:
      case syntax.BinaryExpr():
        dst = self.builder.new_tmp()
        self.builder.emit(Copy(dst, self.visit_expr(node.lhs)))
        self.builder.emit(BinOp(dst, node.bop, self.visit_expr(node.rhs)))
        return dst

      case syntax.Name():
        if not (val := self.builder.get_name(node.name)):
          raise SyntaxError(f"unknown name: {node.name!r}")

        return val

      case syntax.ScalarLiteral():
        val = int(node.text)
        if not -2147483648 <= val <= 2147483647:
          raise SyntaxError(f"invalid integer size: {val}")

        return ImmValue(val)

      case _:
        assert False


class DatapackRenderer:
  def __init__(self, insts: list[IR], ns: str):
    self.insts = insts
    self.ns = ns

    self.funcs: dict[str, list[str]] = {}

  def emit(self, cmd: str, func: str):
    self.funcs.setdefault(func, []).append(cmd)

  def render(self) -> dict[str, list[str]]:
    self.emit(f"scoreboard objectives add {self.ns} dummy", func="setup")

    for inst in self.insts:
      self.render_inst(inst)

    return self.funcs

  def render_inst(self, inst: IR):
    match inst:
      case Copy():
        self.render_copy(inst)

      case BinOp():
        self.render_binop(inst)

      case _:
        assert False

  def render_copy(self, inst: Copy):
    dst, src = inst.dst, inst.src

    match dst, src:
      case ScoreSlot(), ScoreSlot():
        self.emit(
          f"scoreboard players operation "
          f"{dst.holder} {self.ns} = {src.holder} {self.ns}",
          func="main"
        )

      case ScoreSlot(), ImmValue():
        self.emit(
          f"scoreboard players set {dst.holder} {self.ns} {src.val}",
          func="main"
        )

      case _:
        assert False

  def render_binop(self, inst: BinOp):
    dst, bop, src = inst.dst, inst.bop, inst.src

    match dst, bop, src:
      case ScoreSlot(), syntax.BinaryOp.ADD, ImmValue():
        self.emit(
          f"scoreboard players add {dst.holder} {self.ns} {src.val}",
          func="main"
        )

      case ScoreSlot(), syntax.BinaryOp.SUB, ImmValue():
        self.emit(
          f"scoreborad players remove {dst.holder} {self.ns} {src.val}",
          func="main"
        )

      case ScoreSlot(), _, ScoreSlot() | ImmValue():
        op = {
          syntax.BinaryOp.ADD: "+=",
          syntax.BinaryOp.SUB: "-=",
          syntax.BinaryOp.MUL: "*=",
          syntax.BinaryOp.DIV: "/=",
        }[bop]

        if isinstance(src, ImmValue):
          src_holder = str(src.val)
          self.emit(
            f"scoreboard players set {src_holder} {self.ns} {src.val}",
            func="setup"
          )
        else:
          src_holder = src.holder

        self.emit(
          f"scoreboard players operation "
          f"{dst.holder} {self.ns} {op} {src_holder} {self.ns}",
          func="main"
        )

      case _:
        assert False

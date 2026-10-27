from __future__ import annotations

import re

from collections.abc import Iterable, Callable
from dataclasses import dataclass, field
from enum import auto
from functools import cached_property
from typing import Self

from cmdium.utils import DataEnum
from cmdium import error
from cmdium import semantic


class TokenKind(DataEnum):
  INT = auto()
  NAME = auto()
  KW = auto()
  OP = auto()
  EOF = auto()


@dataclass(frozen=True)
class Token:
  kind: TokenKind
  text: str
  file: str
  pos: int

  @cached_property
  def span(self) -> error.Span:
    return error.Span(self.file, self.pos, self.pos + len(self.text))


keywords = {
  "type", "struct",
}

tokens_pat = re.compile(r"""
    (?P<_>    \s+)
  | (?P<INT>  \d+)
  | (?P<NAME> \w+)
  | (?P<OP>   \+ | - | \* | / | % | = | : | ; | \. | , | \( | \) | \{ | \} )
""", re.X)


def lex(src: str, file: str):
  pos, n = 0, len(src)

  while pos < n:
    match = tokens_pat.match(src, pos)
    if not match:
      raise error.CompileError("invalid character", error.Span(file, pos, pos + 1))

    spos = pos
    pos = match.end()
    kind_str = match.lastgroup

    if kind_str == "_":
      continue

    assert kind_str is not None

    kind = {
      "INT": TokenKind.INT,
      "NAME": TokenKind.NAME,
      "OP": TokenKind.OP,
    }[kind_str]

    text = match.group()

    if kind == TokenKind.NAME and text in keywords:
      kind = TokenKind.KW

    yield Token(kind, match.group(), file, spos)

  yield Token(TokenKind.EOF, "", file, pos)


class AST:
  span: error.Span

  def write_span(self, file, spos, epos) -> Self:
    self.span = error.Span(file, spos, epos)
    return self


@dataclass
class CompilationUnit(AST):
  body: list[Statement]


class Statement(AST):
  ...


@dataclass
class VarDecl(Statement):
  name: str
  type: TypeExpr
  init: Expression | None

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class TypeDecl(Statement):
  name: str
  type: TypeExpr


@dataclass
class StructDecl(Statement):
  name: str
  fields: list[StructField]


@dataclass
class Assign(Statement):
  dst: Expression
  val: Expression


class Expression(AST):
  type: semantic.TypeInfo
  lval: bool

  def set_typeinfo(self, type: semantic.TypeInfo, lval: bool):
    self.type = type
    self.lval = lval


class BinaryOp(DataEnum):
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()
  MOD = auto()


@dataclass
class BinaryOpNode(AST):
  bop: BinaryOp


@dataclass
class BinaryExpr(Expression):
  lhs: Expression
  bop: BinaryOpNode
  rhs: Expression


@dataclass
class Name(Expression):
  name: str

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class MemberAccess(Expression):
  src: Expression
  att: str


@dataclass
class ScalarLiteral(Expression):
  text: str


class TypeExpr(AST):
  ...


@dataclass
class NamedType(TypeExpr):
  name: str


@dataclass
class StructField(AST):
  name: str
  type: TypeExpr


class Parser:
  def __init__(self, lexer: Iterable[Token], file: str):
    self.toks = list(lexer)
    self.file = file
    self.pos: int = 0

  def tell(self) -> int:
    return self.pos

  def seek(self, i: int):
    self.pos = i

  def current_tok(self):
    return self.toks[self.pos]

  def curr_spos(self) -> int:
    return self.current_tok().pos

  def prev_epos(self) -> int:
    t = self.toks[self.pos - 1]
    return t.pos + len(t.text)

  def accept(
    self, kind: TokenKind, text: str | None = None
  ) -> Token | None:
    tok = self.toks[self.pos]
    if tok.kind == kind and (text is None or tok.text == text):
      if kind is not TokenKind.EOF:
        self.pos += 1
      return tok
    return None

  def expect(
    self, msg: str, kind: TokenKind, text: str | None = None
  ) -> Token:
    if not (tok := self.accept(kind, text)):
      raise error.CompileError(
        f"expected {msg}", self.current_tok().span
      )
    return tok

  def expect_rule[T: AST](self, msg: str, rule: Callable[[], T | None]) -> T:
    if not (tree := rule()):
      raise error.CompileError(
        f"expected {msg}", self.current_tok().span
      )
    return tree

  def parse(self) -> CompilationUnit:
    # program : stmt* &&EOF
    body: list[Statement] = []
    spos = self.curr_spos()
    
    while stmt := self.parse_stmt():
      body.append(stmt)

    self.expect("statement or end of file", TokenKind.EOF)
    return CompilationUnit(body).write_span(self.file, spos, self.prev_epos())

  def parse_stmt(self) -> Statement | None:
    # stmt : var_decl
    #      | type_decl
    #      | assign

    if decl := self.parse_var_decl():
      return decl

    if type_decl := self.parse_type_decl():
      return type_decl

    if assign := self.parse_assign():
      return assign

    return None

  def parse_var_decl(self) -> VarDecl | None:
    # var_decl : NAME ":" ~ type ("=" expr)? ";"
    mark = self.tell()
    spos = self.curr_spos()

    if (
      (name := self.accept(TokenKind.NAME))
      and self.accept(TokenKind.OP, ":")
    ):
      type = self.expect_rule("type expression", self.parse_type)
        
      if self.accept(TokenKind.OP, "="):
        init = self.expect_rule("expression", self.parse_expr)
        sm_msg = "expr or ';'"
      else:
        init = None
        sm_msg = "';'"

      self.expect(sm_msg, TokenKind.OP, ";")

      return VarDecl(name.text, type, init).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    return None

  def parse_type_decl(self) -> TypeDecl | StructDecl | None:
    # type_decl : "type" ~ NAME "=" type ";"
    #           | "struct" ~ NAME "{" struct_fields? &&"}" ";"
  
    mark = self.tell()
    spos = self.curr_spos()

    if self.accept(TokenKind.KW, "type"):
      name = self.expect("name", TokenKind.NAME)
      self.expect("'='", TokenKind.OP, "=")
      type = self.expect_rule("type expression", self.parse_type)
      self.expect("';'", TokenKind.OP, ";")
      return TypeDecl(name.text, type).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    if self.accept(TokenKind.KW, "struct"):
      name = self.expect("name", TokenKind.NAME)
      self.expect("'{'", TokenKind.OP, "{")

      fields = self.parse_struct_fields()
      cb_msg = "'}'" if fields else "struct fields or '}'"

      self.expect(cb_msg, TokenKind.OP, "}")
      self.expect("';'", TokenKind.OP, ";")
      return StructDecl(name.text, fields).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    return None

  def parse_type(self) -> TypeExpr | None:
    # type : NAME
    spos = self.curr_spos()

    if name := self.accept(TokenKind.NAME):
      return NamedType(name.text).write_span(
        self.file, spos, self.prev_epos()
      )

    return None

  def parse_struct_fields(self) -> list[StructField]:
    # struct_fields : struct_field "," struct_fields
    #               | struct_field ","?
    fields = []
    while f := self.parse_struct_field():
      fields.append(f)
      if not self.accept(TokenKind.OP, ","):
        break
    
    return fields

  def parse_struct_field(self) -> StructField | None:
    # struct_field : NAME ":" ~ type
    mark = self.tell()
    spos = self.curr_spos()

    if (
      (name := self.accept(TokenKind.NAME))
      and self.accept(TokenKind.OP, ":")
    ):
      type = self.expect_rule("type expression", self.parse_type)
      return StructField(name.text, type).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    return None

  def parse_assign(self) -> Assign | None:
    # assign : expr "=" ~ expr ";"
    mark = self.tell()
    spos = self.curr_spos()

    if (
      (dst := self.parse_expr())
      and self.accept(TokenKind.OP, "=")
    ):
      val = self.expect_rule("expression", self.parse_expr)
      self.expect("';'", TokenKind.OP, ";")
      return Assign(dst, val).write_span(self.file, spos, self.prev_epos())

    self.seek(mark)
    return None

  def parse_expr(self) -> Expression | None:
    # expr : term (("+" | "-") &&term)*
    mark = self.tell()
    spos = self.curr_spos()

    if not (lhs := self.parse_term()):
      self.seek(mark)
      return None

    while True:
      mark = self.tell()

      if tok := self.accept(TokenKind.OP, "+"):
        op = BinaryOpNode(BinaryOp.ADD).write_span(*tok.span)

      elif tok := self.accept(TokenKind.OP, "-"):
        op = BinaryOpNode(BinaryOp.SUB).write_span(*tok.span)

      else:
        break

      rhs = self.expect_rule("term expression", self.parse_term)
      lhs = BinaryExpr(lhs, op, rhs).write_span(
        self.file, spos, self.prev_epos()
      )

    return lhs

  def parse_term(self) -> Expression | None:
    # term : primary (("*" | "/" | "%") &&primary)*
    mark = self.tell()
    spos = self.curr_spos()

    if not (lhs := self.parse_primary()):
      self.seek(mark)
      return None

    while True:
      mark = self.tell()

      if tok := self.accept(TokenKind.OP, "*"):
        op = BinaryOpNode(BinaryOp.MUL).write_span(*tok.span)
      
      elif tok := self.accept(TokenKind.OP, "/"):
        op = BinaryOpNode(BinaryOp.DIV).write_span(*tok.span)

      elif tok := self.accept(TokenKind.OP, "%"):
        op = BinaryOpNode(BinaryOp.MOD).write_span(*tok.span)
      
      else:
        break

      rhs = self.expect_rule("primary expression", self.parse_primary)
      lhs = BinaryExpr(lhs, op, rhs).write_span(
        self.file, spos, self.prev_epos()
      )

    return lhs

  def parse_primary(self) -> Expression | None:
    # primary : atom ("." &&NAME)*

    spos = self.curr_spos()

    if not (val := self.parse_atom()):
      return None
    
    while self.accept(TokenKind.OP, "."):
      att = self.expect("field name", TokenKind.NAME)
      val = MemberAccess(val, att.text).write_span(
        self.file, spos, self.prev_epos()
      )

    return val

  def parse_atom(self) -> Expression | None:
    # atom : NAME
    #      | INT
    #      | "(" ~ expr ")"

    spos = self.curr_spos()

    if name := self.accept(TokenKind.NAME):
      return Name(name.text).write_span(self.file, spos, self.prev_epos())

    if int := self.accept(TokenKind.INT):
      return ScalarLiteral(int.text).write_span(
        self.file, spos, self.prev_epos()
      )

    mark = self.tell()
    if self.accept(TokenKind.OP, "("):
      expr = self.expect_rule("expression", self.parse_expr)
      self.expect("')'", TokenKind.OP, ")")
      return expr

    self.seek(mark)
    return None


def parse(src: str, file: str) -> CompilationUnit:
  lexer = lex(src, file)
  parser = Parser(lexer, file)
  tree = parser.parse()
  return tree


if __name__ == "__main__":
  from pprint import pprint

  file = "../demos/test.cmdium"
  with open(file) as f:
    src = f.read()

  pprint(parse(src, file))

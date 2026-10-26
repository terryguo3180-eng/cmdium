from __future__ import annotations

import re

from collections.abc import Iterable, Callable
from dataclasses import dataclass
from enum import auto

from cmdium.utils import DataEnum


class TokenKind(DataEnum):
  INT = auto()
  NAME = auto()
  KW = auto()
  OP = auto()
  EOF = auto()


@dataclass
class Token:
  kind: TokenKind
  text: str
  pos: int


keywords = {
  "type", "struct",
}

tokens_pat = re.compile(r"""
    (?P<_>    \s+)
  | (?P<INT>  \d+)
  | (?P<NAME> \w+)
  | (?P<OP>   \+ | - | \* | / | % | = | : | ; | , | \( | \) | \{ | \} )
""", re.X)


def lex(src: str):
  pos, n = 0, len(src)

  while pos < n:
    match = tokens_pat.match(src, pos)
    if not match:
      raise SyntaxError(f"unexpected token: {src[pos]}")

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

    yield Token(kind, match.group(), pos)

  yield Token(TokenKind.EOF, "", pos)


class AST:
  ...


@dataclass
class CompilationUnit(AST):
  body: list[Statement]


class Statement:
  ...


@dataclass
class VarDecl(Statement):
  name: str
  type: Type
  init: Expression | None


@dataclass
class TypeDecl(Statement):
  name: str
  type: Type


@dataclass
class StructDecl(Statement):
  name: str
  fields: list[StructField]


@dataclass
class Assign(Statement):
  dst: Expression
  val: Expression


class Expression(AST):
  ...


class BinaryOp(DataEnum):
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()
  MOD = auto()


@dataclass
class BinaryExpr(Expression):
  lhs: Expression
  bop: BinaryOp
  rhs: Expression


@dataclass
class Name(Expression):
  name: str


@dataclass
class MemberAccess(Expression):
  src: Expression
  att: str


@dataclass
class ScalarLiteral(Expression):
  text: str


class Type(AST):
  ...


@dataclass
class NamedType(Type):
  name: str


@dataclass
class StructField(AST):
  name: str
  type: Type


class Parser:
  def __init__(self, lexer: Iterable[Token]):
    self.toks = list(lexer)
    self.pos: int = 0

  def tell(self) -> int:
    return self.pos

  def seek(self, i: int):
    self.pos = i

  def current_tok(self):
    return self.toks[self.pos]

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
      raise SyntaxError(f"{self.current_tok()!r}: expected {msg}")
    return tok

  def expect_rule[T: AST](self, msg: str, rule: Callable[[], T | None]) -> T:
    if not (tree := rule()):
      raise SyntaxError(f"{self.current_tok()!r}: expected {msg}")
    return tree

  def parse(self) -> CompilationUnit:
    # program : stmt* &&EOF
    body: list[Statement] = []

    while stmt := self.parse_stmt():
      body.append(stmt)

    if self.accept(TokenKind.EOF):
      return CompilationUnit(body)

    raise SyntaxError(f"{self.current_tok()!r}: expected EOF")

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
      return VarDecl(name.text, type, init)

    self.seek(mark)
    return None

  def parse_type_decl(self) -> TypeDecl | StructDecl | None:
    # type_decl : "type" ~ NAME "=" type ";"
    #           | "struct" ~ NAME "{" struct_fields? &&"}" ";"
  
    mark = self.tell()
    if self.accept(TokenKind.KW, "type"):
      name = self.expect("name", TokenKind.NAME)
      self.expect("'='", TokenKind.OP, "=")
      type = self.expect_rule("type expression", self.parse_type)
      self.expect("';'", TokenKind.OP, ";")
      return TypeDecl(name.text, type)

    self.seek(mark)
    if self.accept(TokenKind.KW, "struct"):
      name = self.expect("name", TokenKind.NAME)
      self.expect("'{'", TokenKind.OP, "{")

      fields = self.parse_struct_fields()
      cb_msg = "'}'" if fields else "struct fields or '}'"

      self.expect(cb_msg, TokenKind.OP, "}")
      self.expect("';'", TokenKind.OP, ";")
      return StructDecl(name.text, fields)

    self.seek(mark)
    return None

  def parse_type(self) -> Type | None:
    # type : NAME
    if name := self.accept(TokenKind.NAME):
      return NamedType(name.text)

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

    if (
      (name := self.accept(TokenKind.NAME))
      and self.accept(TokenKind.OP, ":")
    ):
      type = self.expect_rule("type expression", self.parse_type)
      return StructField(name.text, type)

    self.seek(mark)
    return None

  def parse_assign(self) -> Assign | None:
    # assign : expr "=" expr ";"
    mark = self.tell()

    if (
      (dst := self.parse_expr())
      and self.accept(TokenKind.OP, "=")
      and (val := self.parse_expr())
      and self.accept(TokenKind.OP, ";")
    ):
      return Assign(dst, val)

    self.seek(mark)
    return None

  def parse_expr(self) -> Expression | None:
    # expr : term (("+" | "-") &&term)*
    mark = self.tell()

    if not (lhs := self.parse_term()):
      self.seek(mark)
      return None

    while True:
      mark = self.tell()

      if self.accept(TokenKind.OP, "+"):
        op = BinaryOp.ADD
      elif self.accept(TokenKind.OP, "-"):
        op = BinaryOp.SUB
      else:
        break

      rhs = self.expect_rule("term expression", self.parse_term)
      lhs = BinaryExpr(lhs, op, rhs)

    return lhs

  def parse_term(self) -> Expression | None:
    # term : primary (("*" | "/" | "%") &&primary)*
    mark = self.tell()

    if not (lhs := self.parse_primary()):
      self.seek(mark)
      return None

    while True:
      mark = self.tell()

      if self.accept(TokenKind.OP, "*"):
        op = BinaryOp.MUL
      elif self.accept(TokenKind.OP, "/"):
        op = BinaryOp.DIV
      elif self.accept(TokenKind.OP, "%"):
        op = BinaryOp.MOD
      else:
        break

      rhs = self.expect_rule("primary expression", self.parse_term)
      lhs = BinaryExpr(lhs, op, rhs)

    return lhs

  def parse_primary(self) -> Expression | None:
    # primary : atom ("." &&NAME)*

    if not (val := self.parse_atom()):
      return None
    
    while self.accept(TokenKind.OP, "."):
      att = self.expect("field name", TokenKind.NAME)
      val = MemberAccess(val, att.text)

    return val

  def parse_atom(self) -> Expression | None:
    # atom : NAME
    #      | INT
    #      | "(" expr ")"

    if name := self.accept(TokenKind.NAME):
      return Name(name.text)

    if int := self.accept(TokenKind.INT):
      return ScalarLiteral(int.text)

    mark = self.tell()
    if (
      self.accept(TokenKind.OP, "(")
      and (expr := self.parse_expr())
      and self.accept(TokenKind.OP, ")")
    ):
      return expr

    self.seek(mark)
    return None


def parse(src: str) -> CompilationUnit:
  lexer = lex(src)
  parser = Parser(lexer)
  tree = parser.parse()
  return tree


if __name__ == "__main__":
  from pprint import pprint

  with open("../demos/test.cmdium") as f:
    src = f.read()

  pprint(parse(src))

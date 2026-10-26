from __future__ import annotations

import re

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto


class DataEnum(Enum):
  def __repr__(self):
    return f"{type(self).__name__}.{self.name}"


class TokenKind(DataEnum):
  INT = auto()
  NAME = auto()
  OP = auto()
  EOF = auto()


@dataclass
class Token:
  kind: TokenKind
  text: str
  pos: int


tokens_pat = re.compile(r"""
    (?P<_>    \s+)
  | (?P<INT>  \d+)
  | (?P<NAME> \w+)
  | (?P<OP>   \+ | - | \* | / | % | = | : | ; | \( | \) )
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

    yield Token(kind, match.group(), pos)

  yield Token(TokenKind.EOF, "", pos)


@dataclass
class AST:
  ...


@dataclass
class CompilationUnit(AST):
  body: list[Statement]


@dataclass
class Statement:
  ...


@dataclass
class VarDecl(Statement):
  name: str
  type: Type
  init: Expression | None


@dataclass
class Assign(Statement):
  dst: LeftValue
  val: Expression


@dataclass
class Expression(AST):
  ...


@dataclass
class LeftValue(Expression):
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
class Name(LeftValue):
  name: str


@dataclass
class ScalarLiteral(Expression):
  text: str


@dataclass
class Type(AST):
  ...


@dataclass
class NamedType(Type):
  name: str



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
  
  def parse(self) -> CompilationUnit:
    # program : stmt* &&EOF
    body: list[Statement] = []

    while stmt := self.parse_stmt():
      body.append(stmt)

    if self.accept(TokenKind.EOF):
      return CompilationUnit(body)

    raise SyntaxError(f"{self.current_tok()!r}: expected EOF")

  def parse_stmt(self) -> Statement | None:
    # stmt : vardecl
    #      | assign
    if decl := self.parse_vardecl():
      return decl

    if assign := self.parse_assign():
      return assign

    return None

  def parse_vardecl(self) -> VarDecl | None:
    # vardecl : NAME ":" type ("=" &&expr)? &&";"
    mark = self.tell()

    if (
      (name := self.accept(TokenKind.NAME))
      and self.accept(TokenKind.OP, ":")
      and (type := self.parse_type())
    ):
      if self.accept(TokenKind.OP, "="):
        init = self.parse_expr()
        if init is None:
          raise SyntaxError(
            f"{self.current_tok()!r}: expected expr"
          )
      else:
        init = None

      if not self.accept(TokenKind.OP, ";"):
        raise SyntaxError(
          f"{self.current_tok()!r}: expected ';'"
        )

      return VarDecl(name.text, type, init)

    self.seek(mark)
    return None

  def parse_type(self) -> Type | None:
    # type : NAME
    if name := self.accept(TokenKind.NAME):
      return NamedType(name.text)

    return None

  def parse_assign(self) -> Assign | None:
    # assign : lval "=" expr ";"
    mark = self.tell()

    if (
      (dst := self.parse_lval())
      and self.accept(TokenKind.OP, "=")
      and (val := self.parse_expr())
      and self.accept(TokenKind.OP, ";")
    ):
      return Assign(dst, val)

    self.seek(mark)
    return None

  def parse_lval(self) -> LeftValue | None:
    # lval : NAME
    if name := self.accept(TokenKind.NAME):
      return Name(name.text)

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

      if rhs := self.parse_term():
        lhs = BinaryExpr(lhs, op, rhs)
      else:
        raise SyntaxError(
          f"{self.current_tok()!r}: expected term"
        )

    return lhs

  def parse_term(self) -> Expression | None:
    # term : atom (("*" | "/" | "%") &&atom)*
    mark = self.tell()

    if not (lhs := self.parse_atom()):
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

      if rhs := self.parse_atom():
        lhs = BinaryExpr(lhs, op, rhs)
      else:
        raise SyntaxError(
          f"{self.current_tok()!r}: expected atom"
        )

    return lhs

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

  if tree is None:
    raise SyntaxError("invalid program")

  return tree


if __name__ == "__main__":
  from pprint import pprint

  with open("../demos/test.cmdium") as f:
    src = f.read()

  pprint(parse(src))

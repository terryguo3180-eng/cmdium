from __future__ import annotations

import re

from collections.abc import Iterable, Callable
from dataclasses import dataclass, field
from enum import auto
from functools import cached_property
from typing import Self

from cmdium import error
from cmdium import semantic
from cmdium import utils


class TokenKind(utils.DataEnum):
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
  "type", "struct", "func", "if", "else", "while", "break", "continue",
  "return", "T", "F",
}

tokens_pat = re.compile(r"""
    (?P<_>    \s+ | \#[^\n]*)
  | (?P<INT>  \d+)
  | (?P<NAME> \w+)
  | (?P<OP>   \+= | -= | \*= | /= | %= | <= | >= | == | != | -> | := |
              \+ | - | \* | / | % | & | \| | = | ! | : | ; | \. | , | < | > |
              \( | \) | \[ | \] | \{ | \} )
""", re.X)


def lex(src: str, file: str):
  pos, n = 0, len(src)

  while pos < n:
    match = tokens_pat.match(src, pos)
    if not match:
      raise error.CompileError(
        "invalid character", error.Span(file, pos, pos + 1)
      )

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

  def write_span(self, file: str, spos: int, epos: int) -> Self:
    self.span = error.Span(file, spos, epos)
    return self

  def __str__(self):
    return utils.dump(self)


@dataclass
class CompilationUnit(AST):
  body: list[Statement]


class StatementBase(AST):
  ...


class RawStatementBase(StatementBase):
  ...


class DeclarationBase(StatementBase):
  ...


@dataclass
class VarDecl(DeclarationBase):
  name: str
  type: TypeExpr
  init: Expression | None
  name_tok: Token

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class TypeDecl(DeclarationBase):
  name: str
  type: TypeExpr
  name_tok: Token


@dataclass
class StructDecl(DeclarationBase):
  name: str
  fields: list[StructField]
  name_tok: Token


@dataclass
class FuncDecl(DeclarationBase):
  name: str
  params: list[FuncParam]
  ret_type: TypeExpr | None
  body: Block
  name_tok: Token

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class FuncParam(AST):
  name: str
  type: TypeExpr
  name_tok: Token

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class If(RawStatementBase):
  test: Expression
  body: RawStatement
  orelse: RawStatement | None


@dataclass
class While(RawStatementBase):
  test: Expression | None
  body: RawStatement


@dataclass
class Block(RawStatementBase):
  body: list[Statement]


@dataclass
class Break(RawStatementBase):
  ...


@dataclass
class Continue(RawStatementBase):
  ...


@dataclass
class Return(RawStatementBase):
  val: Expression | None


@dataclass
class Pass(RawStatementBase):
  ...


@dataclass
class Assign(RawStatementBase):
  dst: Expression
  val: Expression


@dataclass
class AugAssign(RawStatementBase):
  dst: Expression
  bop: BinaryOpNode
  val: Expression


@dataclass
class ExprStmt(RawStatementBase):
  val: Expression


class ExpressionBase(AST):
  type: semantic.TypeInfo
  lval: bool

  def set_typeinfo(self, type: semantic.TypeInfo, lval: bool):
    self.type = type
    self.lval = lval


class BinaryOp(utils.DataEnum):
  ADD = auto()  # +
  SUB = auto()  # -
  MUL = auto()  # *
  DIV = auto()  # /
  MOD = auto()  # %
  
  OR = auto()   # |
  AND = auto()  # &

  CLT = auto()  # <
  CGT = auto()  # >
  CLE = auto()  # <=
  CGE = auto()  # >=
  CEQ = auto()  # ==
  CNE = auto()  # !=


@dataclass
class BinaryOpNode(AST):
  bop: BinaryOp


@dataclass
class BinaryExpr(ExpressionBase):
  lhs: Expression
  bop: BinaryOpNode
  rhs: Expression


class UnaryOp(utils.DataEnum):
  NEG = auto()
  NOT = auto()


@dataclass
class UnaryOpNode(AST):
  uop: UnaryOp


@dataclass
class UnaryExpr(ExpressionBase):
  uop: UnaryOpNode
  val: Expression


@dataclass
class Name(ExpressionBase):
  name: str

  sym: semantic.Symbol = field(init=False, repr=False)

  def set_symbol(self, sym: semantic.Symbol):
    self.sym = sym


@dataclass
class MemberAccess(ExpressionBase):
  src: Expression
  att: str


@dataclass
class Call(ExpressionBase):
  src: Expression
  args: list[Expression]


class ScalarKind(utils.DataEnum):
  I32 = auto()
  BOOL = auto()


@dataclass
class ScalarLiteral(ExpressionBase):
  text: str
  kind: ScalarKind


class TypeExprBase(AST):
  ...


@dataclass
class NamedType(TypeExprBase):
  name: str


@dataclass
class StructField(AST):
  name: str
  type: TypeExpr

  name_tok: Token


@dataclass
class FuncType(TypeExprBase):
  params: list[TypeExpr]
  ret_type: TypeExpr


Declaration = VarDecl | TypeDecl | StructDecl | FuncDecl
RawStatement = (
  If | While | Block | Break | Continue | Return | Pass | Assign | AugAssign |
  ExprStmt
)
Statement = Declaration | RawStatement
Expression = (
  BinaryExpr | UnaryExpr | Name | MemberAccess | Call | ScalarLiteral
)
TypeExpr = NamedType | FuncType


class Parser:
  def __init__(self, lexer: Iterable[Token], file: str):
    self.toks = list(lexer)
    self.file = file
    self.pos: int = 0

    self.expr_precedence: dict[str, tuple[int, BinaryOp]] = {
      "|": (0, BinaryOp.OR),
      "&": (1, BinaryOp.AND),
      "==": (2, BinaryOp.CEQ), "!=": (2, BinaryOp.CNE),
      "<": (3, BinaryOp.CLT), ">": (3, BinaryOp.CGT),
      "<=": (3, BinaryOp.CLE), ">=": (3, BinaryOp.CGE),
      "+": (4, BinaryOp.ADD), "-": (4, BinaryOp.SUB),
      "*": (5, BinaryOp.MUL), "/": (5, BinaryOp.DIV), "%": (5, BinaryOp.MOD),
    }
    self.augassign_ops: dict[str, BinaryOp] = {
      "+=": BinaryOp.ADD,
      "-=": BinaryOp.SUB,
      "*=": BinaryOp.MUL,
      "/=": BinaryOp.DIV,
      "%=": BinaryOp.MOD,
    }

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
    # stmt : decl
    #      | raw_stmt
    
    if decl := self.parse_decl():
      return decl

    if stmt := self.parse_raw_stmt():
      return stmt

    return None

  def parse_raw_stmt(self) -> RawStatement | None:
    # raw_stmt : block
    #          | if
    #          | while
    #          | "break" ~ ";"
    #          | "continue" ~ ";"
    #          | "return" ~ expr ";"
    #          | assign
    #          | aug_assign
    #          | expr ~ ";"
    #          | ";"

    spos = self.curr_spos()

    if block := self.parse_block():
      return block

    if if_ := self.parse_if():
      return if_

    if while_ := self.parse_while():
      return while_

    if self.accept(TokenKind.KW, "break"):
      self.expect("';'", TokenKind.OP, ";")
      return Break().write_span(self.file, spos, self.prev_epos())

    if self.accept(TokenKind.KW, "continue"):
      self.expect("';'", TokenKind.OP, ";")
      return Continue().write_span(self.file, spos, self.prev_epos())

    if self.accept(TokenKind.KW, "return"):
      expr = self.parse_expr()
      self.expect("';'" if expr else "expression or ';'", TokenKind.OP, ";")
      return Return(expr).write_span(self.file, spos, self.prev_epos())

    if assign := self.parse_assign():
      return assign

    if aug := self.parse_aug_assign():
      return aug

    if val := self.parse_expr():
      self.expect("';'", TokenKind.OP, ";")
      return ExprStmt(val).write_span(self.file, spos, self.prev_epos())

    if tok := self.accept(TokenKind.OP, ";"):
      return Pass().write_span(*tok.span)

    return None

  def parse_block(self):
    # block : "{" ~ stmt* "}"
    spos = self.curr_spos()

    if self.accept(TokenKind.OP, "{"):
      body: list[Statement] = []
      while stmt := self.parse_stmt():
        body.append(stmt)

      self.expect("'}'", TokenKind.OP, "}")
      return Block(body).write_span(self.file, spos, self.prev_epos())

    return None

  def parse_if(self):
    # if : "if" ~ "(" expr ")" raw_stmt ("else" ~ raw_stmt)?
    spos = self.curr_spos()

    if self.accept(TokenKind.KW, "if"):
      self.expect("'('", TokenKind.OP, "(")
      test = self.expect_rule("expression", self.parse_expr)
      self.expect("')'", TokenKind.OP, ")")
      body = self.expect_rule(
        "statement other than declaration", self.parse_raw_stmt
      )
      if self.accept(TokenKind.KW, "else"):
        orelse = self.expect_rule(
          "statement other than declaration", self.parse_raw_stmt
        )
      else:
        orelse = None

      return If(test, body, orelse).write_span(
        self.file, spos, self.prev_epos()
      )

    return None

  def parse_while(self):
    # while : "while" ~ ("(" ~ expr ")")? raw_stmt
    spos = self.curr_spos()

    if self.accept(TokenKind.KW, "while"):
      if self.accept(TokenKind.OP, "("):
        test = self.expect_rule("expression", self.parse_expr)
        self.expect("')'", TokenKind.OP, ")")
        body_msg = "statement other than declaration"
      else:
        test = None
        body_msg = "'(' or statement other than declaration"

      body = self.expect_rule(body_msg, self.parse_raw_stmt)

      return While(test, body).write_span(self.file, spos, self.prev_epos())

    return None

  def parse_decl(self) -> Declaration | None:
    # decl : var_decl
    #      | type_decl
    #      | func_decl
    
    if var_decl := self.parse_var_decl():
      return var_decl

    if type_decl := self.parse_type_decl():
      return type_decl

    if func_decl := self.parse_func_decl():
      return func_decl

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

      return VarDecl(name.text, type, init, name).write_span(
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
      return TypeDecl(name.text, type, name).write_span(
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
      return StructDecl(name.text, fields, name).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    return None

  def parse_func_decl(self) -> FuncDecl | None:
    # func_decl : "func" ~ NAME "(" params? ~ ")" ("->" ~ type)? block

    spos = self.curr_spos()

    if not self.accept(TokenKind.KW, "func"):
      return None

    name = self.expect("function name", TokenKind.NAME)
    self.expect("'('", TokenKind.OP, "(")

    params = self.parse_params() or []
    self.expect("')'" if params else "parameters or ')'", TokenKind.OP, ")")

    if self.accept(TokenKind.OP, "->"):
      ret_type = self.expect_rule("type expression", self.parse_type)
      body_msg = "block"
    else:
      ret_type = None
      body_msg = "'->' or block"

    body = self.expect_rule(body_msg, self.parse_block)
    return FuncDecl(name.text, params, ret_type, body, name).write_span(
      self.file, spos, self.prev_epos()
    )

  def parse_params(self) -> list[FuncParam] | None:
    # params : param ("," param)* ","?
    params: list[FuncParam] = []
    while param := self.parse_param():
      params.append(param)
      if not self.accept(TokenKind.OP, ","):
        break

    return params

  def parse_param(self) -> FuncParam | None:
    # param : NAME ~ ":" type

    spos = self.curr_spos()

    if not (name := self.accept(TokenKind.NAME)):
      return None

    self.expect("':'", TokenKind.OP, ":")
    type = self.expect_rule("type expression", self.parse_type)

    return FuncParam(name.text, type, name).write_span(
      self.file, spos, self.prev_epos()
    )

  def parse_type(self) -> TypeExpr | None:
    # type : NAME
    #      | "(" (type ("," type)*)? ~ ")" "->" type
    #      | "(" type ~ ")"

    spos = self.curr_spos()

    if name := self.accept(TokenKind.NAME):
      return NamedType(name.text).write_span(
        self.file, spos, self.prev_epos()
      )

    if self.accept(TokenKind.OP, "("):
      params: list[TypeExpr] = []
      comma = False
      while type := self.parse_type():
        params.append(type)
        if not self.accept(TokenKind.OP, ","):
          break
        else:
          comma = True

      self.expect(
        "type expression or ')'" if comma else "',' or ')'",
        TokenKind.OP, ")"
      )

      if comma:
        self.expect("'->'", TokenKind.OP, "->")
        type = self.expect_rule("type expression", self.parse_type)
        return FuncType(params, type).write_span(
          self.file, spos, self.prev_epos()
        )

      if self.accept(TokenKind.OP, "->") and (type := self.parse_type()):
        return FuncType(params, type).write_span(
          self.file, spos, self.prev_epos()
        )

      return params[0]  # grouped type

    return None

  def parse_struct_fields(self) -> list[StructField]:
    # struct_fields : struct_field ("," struct_field)* ","?
    fields: list[StructField] = []
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
      return StructField(name.text, type, name).write_span(
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

  def parse_aug_assign(self) -> AugAssign | None:
    # aug_assign : expr augop ~ expr ";"
    mark = self.tell()
    spos = self.curr_spos()

    if (
      (dst := self.parse_expr())
      and (bop := self.parse_augop())
    ):
      val = self.expect_rule("expression", self.parse_expr)
      self.expect("';'", TokenKind.OP, ";")
      return AugAssign(dst, bop, val).write_span(
        self.file, spos, self.prev_epos()
      )

    self.seek(mark)
    return None

  def parse_augop(self) -> BinaryOpNode | None:
    # augop : "+=" | "-=" | "*=" | "/=" | "%="
    for s, op in self.augassign_ops.items():
      if tok := self.accept(TokenKind.OP, s):
        return BinaryOpNode(op).write_span(*tok.span)

    return None

  def parse_expr(self, min_prec: int = 0) -> Expression | None:
    spos = self.curr_spos()

    if not (lhs := self.parse_unary()):
      return None

    while True:
      tok = self.current_tok()
      entry = (
        self.expr_precedence.get(tok.text)
        if tok.kind == TokenKind.OP
        else None
      )
      if entry is None or entry[0] < min_prec:
        break

      prec, op = entry
      self.accept(TokenKind.OP, tok.text)
      bop = BinaryOpNode(op).write_span(*tok.span)

      rhs = self.expect_rule("expression", lambda: self.parse_expr(prec + 1))
      lhs = BinaryExpr(lhs, bop, rhs).write_span(
        self.file, spos, self.prev_epos()
      )

    return lhs

  def parse_unary(self) -> Expression | None:
    # unary : ("-" | "!") ~ unary
    #       | primary
    spos = self.curr_spos()

    if op := self.accept(TokenKind.OP, "-"):
      val = self.expect_rule("unary expression", self.parse_unary)
      uop = UnaryOpNode(UnaryOp.NEG).write_span(*op.span)
      return UnaryExpr(uop, val).write_span(self.file, spos, self.prev_epos())

    if op := self.accept(TokenKind.OP, "!"):
      val = self.expect_rule("unary expression", self.parse_unary)
      uop = UnaryOpNode(UnaryOp.NOT).write_span(*op.span)
      return UnaryExpr(uop, val).write_span(self.file, spos, self.prev_epos())

    if primary := self.parse_primary():
      return primary

    return None

  def parse_primary(self) -> Expression | None:
    # primary : atom ("." ~ NAME | "(" args? ~ ")")*
    spos = self.curr_spos()
    if not (val := self.parse_atom()):
      return None

    while True:
      if self.accept(TokenKind.OP, "."):
        att = self.expect("field name", TokenKind.NAME)
        val = MemberAccess(val, att.text).write_span(
          self.file, spos, self.prev_epos()
        )
      elif self.accept(TokenKind.OP, "("):
        args = self.parse_args() or []
        self.expect(
          "')'" if args else "arguments or ')'", TokenKind.OP, ")"
        )
        val = Call(val, args).write_span(self.file, spos, self.prev_epos())
      else:
        return val

  def parse_args(self) -> list[Expression] | None:
    # args : expr ("," expr)* ","?
    args: list[Expression] = []
    while arg := self.parse_expr():
      args.append(arg)
      if not self.accept(TokenKind.OP, ","):
        break

    return args

  def parse_atom(self) -> Expression | None:
    # atom : NAME
    #      | INT
    #      | "T" | "F"
    #      | "(" ~ expr ")"

    spos = self.curr_spos()

    if name := self.accept(TokenKind.NAME):
      return Name(name.text).write_span(self.file, spos, self.prev_epos())

    if tok := self.accept(TokenKind.INT):
      return ScalarLiteral(tok.text, ScalarKind.I32).write_span(
        self.file, spos, self.prev_epos()
      )

    if tok := self.accept(TokenKind.KW, "T"):
      return ScalarLiteral("T", ScalarKind.BOOL).write_span(
        self.file, spos, self.prev_epos()
      )

    if tok := self.accept(TokenKind.KW, "F"):
      return ScalarLiteral("F", ScalarKind.BOOL).write_span(
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
  file = "../demos/test.cmdium"
  with open(file) as f:
    src = f.read()

  print(parse(src, file))
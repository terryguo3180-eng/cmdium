from __future__ import annotations

import json

from contextlib import contextmanager
from dataclasses import dataclass
from enum import auto
from typing import overload

from cmdium import lowering
from cmdium import optimize
from cmdium.utils import DataEnum



@dataclass(frozen=True)
class ScoreVar:
  holder: str
  objective: str

  def __str__(self):
    return f"{self.holder} {self.objective}"


@dataclass(frozen=True)
class ScoreImm:
  val: int

  def __str__(self):
    return str(self.val)


@dataclass(frozen=True)
class ScoreGet:
  var: ScoreVar

  def __str__(self):
    return f"scoreboard players get {self.var!s}"


class NBTContainer:
  ...


@dataclass(frozen=True)
class Storage(NBTContainer):
  ns: str
  id: str

  def get_nbtvar(self, path: str):
    return NBTVar(path, self)

  def __str__(self):
    return f"storage {self.ns}:{self.id}"


@dataclass(frozen=True)
class Entity(NBTContainer):
  selector: str

  def __str__(self):
    return f"entity {self.selector}"


@dataclass(frozen=True)
class Block(NBTContainer):
  coord: tuple[str, str, str]

  def __str__(self):
    x, y, z = self.coord
    return f"block {x} {y} {z}"


@dataclass(frozen=True)
class NBTVar:
  path: str
  container: NBTContainer

  def __str__(self):
    return f"{self.container!s} {self.path}"


@dataclass(frozen=True)
class NBTGet:
  var: NBTVar

  def __str__(self):
    return f"data get {self.var!s}"


@dataclass(frozen=True)
class NBTImm:
  val: str

  def __str__(self):
    return self.val


@dataclass(frozen=True)
class MacroImm:
  id: str

  def __str__(self):
    return f"$({self.id})"


CommandVar = ScoreVar | NBTVar
CommandImm = ScoreImm | NBTImm | MacroImm
CommandOp = CommandVar | CommandImm


class ScoreOp(DataEnum):
  SET = auto()
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()
  MOD = auto()
  CGT = auto()
  CLT = auto()
  SWP = auto()



@dataclass(frozen=True)
class Scoreboard:
  dst: ScoreVar
  sop: ScoreOp
  src: ScoreVar | ScoreImm | MacroImm

  def __str__(self):
    prefix = "$"[:isinstance(self.src, MacroImm)]

    match self.sop:
      case ScoreOp.SET | ScoreOp.ADD | ScoreOp.SUB as sop if (
        isinstance(self.src, ScoreImm)
      ):
        op = {
          ScoreOp.SET: "set", ScoreOp.ADD: "add", ScoreOp.SUB: "remove"
        }[sop]
        if self.src.val < 0 and op != "set":
          op = ("add", "remove")[op == "add"]
        val = abs(self.src.val)

        return f"{prefix}scoreboard players {op} {self.dst!s} {val}"

      case sop:
        op = {
          ScoreOp.SET: "=",
          ScoreOp.ADD: "+=",
          ScoreOp.SUB: "-=",
          ScoreOp.MUL: "*=",
          ScoreOp.DIV: "/=",
          ScoreOp.MOD: "%=",
          ScoreOp.CGT: ">",
          ScoreOp.CLT: "<",
          ScoreOp.SWP: "><",
        }[sop]
        return (
          prefix
          + f"scoreboard players operation {self.dst!s} {op} {self.src!s}"
        )


@dataclass(frozen=True)
class DataModify:
  dst: NBTVar
  src: NBTVar | NBTImm

  def __str__(self):
    match self.src:
      case NBTVar() as src:
        return f"data modify {self.dst!s} set from {src!s}"

      case NBTImm() as src:
        return f"data modify {self.dst!s} set value {src!s}"


@dataclass(frozen=True)
class DataRemove:
  var: NBTVar

  def __str__(self):
    return f"data remove {self.var!s}"


class NBTDtype(DataEnum):
  BYTE = auto()
  SHORT = auto()
  INT = auto()
  LONG = auto()
  FLOAT = auto()
  DOUBLE = auto()

  def __str__(self):
    return self.name.lower()


class StoreMode(DataEnum):
  RESULT = auto()
  SUCCESS = auto()

  def __str__(self):
    return self.name.lower()


class CompareOp(DataEnum):
  CEQ = auto()
  CNE = auto()
  CLT = auto()
  CGT = auto()
  CLE = auto()
  CGE = auto()


@dataclass(frozen=True)
class IfScore:
  lhs: ScoreVar | ScoreImm
  cop: CompareOp
  rhs: ScoreVar | ScoreImm
  neg: bool = False

  @classmethod
  def of(
    cls,
    lhs: ScoreVar | ScoreImm,
    cop: CompareOp,
    rhs: ScoreVar | ScoreImm,
  ) -> IfScore:
    if isinstance(lhs, ScoreVar):
      return cls(lhs, cop, rhs)
    if isinstance(rhs, ScoreVar):
      cop = {
        CompareOp.CEQ: CompareOp.CEQ,
        CompareOp.CNE: CompareOp.CNE,
        CompareOp.CLT: CompareOp.CGT,
        CompareOp.CLE: CompareOp.CGE,
        CompareOp.CGT: CompareOp.CLT,
        CompareOp.CGE: CompareOp.CLE,
      }[cop]
      return cls(rhs, cop, lhs)
    raise NotImplementedError

  def __str__(self):
    cop = self.cop
    rhs = self.rhs

    match rhs:
      case ScoreVar():
        match self.cop:
          case (
            CompareOp.CEQ | CompareOp.CNE |
            CompareOp.CLT | CompareOp.CLE |
            CompareOp.CGT | CompareOp.CGE
          ):
            op, neg = {
              CompareOp.CEQ: ("=",  False),
              CompareOp.CNE: ("=",  True),
              CompareOp.CLT: ("<",  False),
              CompareOp.CLE: ("<=", False),
              CompareOp.CGT: (">",  False),
              CompareOp.CGE: (">=", False),
            }[cop]
            cond = ("if", "unless")[neg != self.neg]
            return f"{cond} score {self.lhs!s} {op} {rhs!s}"

      case ScoreImm(val=val):
        match cop:
          case CompareOp.CNE | CompareOp.CEQ:
            cond = ("if", "unless")[(cop == CompareOp.CNE) != self.neg]
            lfix = rfix = ""

          case CompareOp.CLT | CompareOp.CLE:
            if cop == CompareOp.CLT:
              val -= 1
            cond = ("if", "unless")[self.neg]
            lfix = ".."
            rfix = ""

          case CompareOp.CGT | CompareOp.CGE:
            if cop == CompareOp.CGT:
              val += 1
            cond = ("if", "unless")[self.neg]
            lfix = ""
            rfix = ".."

        return f"{cond} score {self.lhs!s} matches {lfix}{val}{rfix}"


@dataclass(frozen=True)
class IfNBT:
  var: NBTVar
  val: NBTImm
  neg: bool = False

  @classmethod
  def try_of(
    cls, var: NBTVar, val: NBTImm, neg: bool = False
  ) -> IfNBT | None:
    if var.path.endswith("]"):  # index
      return None
    return cls(var, val, neg)

  def __str__(self):
    par, _, stem = self.var.path.rpartition(".")
    cond = ("if", "unless")[self.neg]
    return f"{cond} data {self.var.container!s} {par}{{{stem}:{self.val}}}"


@dataclass(frozen=True)
class StoreScore:
  dst: ScoreVar
  mode: StoreMode

  def __str__(self):
    return f"store {self.mode!s} score {self.dst!s}"


@dataclass(frozen=True)
class StoreNBT:
  dst: NBTVar
  mode: StoreMode
  dtype: NBTDtype
  scale: float

  def __str__(self):
    return f"{self.mode!s} {self.dst!s} {self.dtype!s} {self.scale}"


@dataclass(frozen=True)
class Execute:
  clauses: tuple[ExecuteClause, ...]
  cmd: Command | None

  def __str__(self):
    if self.cmd is None:
      run = cmd = ""
    elif isinstance(self.cmd, ExecuteClause):
      run = ""
      cmd = str(self.cmd).removeprefix("execute ")
    else:
      run = " run "
      cmd = str(self.cmd)

    prefix = "$"[:cmd.startswith("$")]
    cmd = cmd.lstrip("$")

    return (
      prefix + "execute " + " ".join(str(c) for c in self.clauses) + run + cmd
    )


@dataclass(frozen=True)
class Call:
  ns: str
  func: NBTImm | MacroImm

  def __str__(self):
    prefix = "$"[:isinstance(self.func, MacroImm)]
    if isinstance(self.func, NBTImm):
      func = self.func.val[1:-1]
    else:
      func = str(self.func)
    return f"{prefix}function {self.ns}:{func}"


@dataclass(frozen=True)
class ReturnVal:
  val: int

  def __str__(self):
    return f"return {self.val}"


@dataclass(frozen=True)
class ReturnCmd:
  cmd: Command

  def __str__(self):
    cmd = str(self.cmd)
    prefix = "$"[:cmd.startswith("$")]
    cmd = cmd.lstrip("$")
    return f"{prefix}return run {cmd}"


@dataclass(frozen=True)
class PlainCommand:
  text: str

  def __str__(self):
    return self.text


ExecuteClause = IfScore | IfNBT | StoreScore | StoreNBT

Command = (
  Scoreboard | DataModify | DataRemove | Call | PlainCommand | ScoreGet |
  NBTGet | Execute | ReturnVal | ReturnCmd
)


@dataclass
class MCFunction:
  name: str
  cmds: list[Command]
  
  def emit(self, cmd: Command):
    self.cmds.append(cmd)


def _classify(bop: lowering.BinaryOp) -> ScoreOp | CompareOp:
  match bop:
    case lowering.BinaryOp.ADD: return ScoreOp.ADD
    case lowering.BinaryOp.SUB: return ScoreOp.SUB
    case lowering.BinaryOp.MUL: return ScoreOp.MUL
    case lowering.BinaryOp.DIV: return ScoreOp.DIV
    case lowering.BinaryOp.MOD: return ScoreOp.MOD
    case lowering.BinaryOp.OR: return ScoreOp.CGT
    case lowering.BinaryOp.AND: return ScoreOp.CLT
    case lowering.BinaryOp.CEQ: return CompareOp.CEQ
    case lowering.BinaryOp.CNE: return CompareOp.CNE
    case lowering.BinaryOp.CLT: return CompareOp.CLT
    case lowering.BinaryOp.CLE: return CompareOp.CLE
    case lowering.BinaryOp.CGT: return CompareOp.CGT
    case lowering.BinaryOp.CGE: return CompareOp.CGE


def _fold(bop: lowering.BinaryOp, a: int, b: int) -> int:
  match bop:
    case lowering.BinaryOp.ADD: return a + b
    case lowering.BinaryOp.SUB: return a - b
    case lowering.BinaryOp.MUL: return a * b
    case lowering.BinaryOp.DIV: return a // b
    case lowering.BinaryOp.MOD: return a % b
    case lowering.BinaryOp.OR: return a | b
    case lowering.BinaryOp.AND: return a & b
    case lowering.BinaryOp.CEQ: return int(a == b)
    case lowering.BinaryOp.CNE: return int(a != b)
    case lowering.BinaryOp.CLT: return int(a < b)
    case lowering.BinaryOp.CLE: return int(a <= b)
    case lowering.BinaryOp.CGT: return int(a > b)
    case lowering.BinaryOp.CGE: return int(a >= b)


class MCFuncBuilder:
  def __init__(
    self,
    name: str,
    ns: str,
    stg: Storage,
    macro_stg: Storage,
    binop_tmp: ScoreVar,
    comp_tmp: NBTVar,
    objective: str,
    setup_func: str,
    consts: set[int],
  ):
    self.ns = ns
    self.stg = stg
    self.macro_stg = macro_stg
    self.binop_tmp = binop_tmp
    self.comp_tmp = comp_tmp
    self.consts = consts
    self.objective = objective
    self.setup_func = setup_func

    self.func = MCFunction(name, [])

  def emit(self, cmd: Command):
    self.func.emit(cmd)

  def pop(self) -> Command:
    return self.func.cmds.pop()

  def add_const(self, const: int):
    self.consts.add(const)

  def const_holder(self, const: int) -> ScoreVar:
    self.add_const(const)
    return ScoreVar(str(const), self.objective)

  def score_holder(
    self, place: lowering.Place | lowering.ImmValue
  ) -> ScoreVar:
    if isinstance(place, lowering.ImmValue):
      return self.const_holder(int(place.val))
    if place.temp:
      return ScoreVar(f"#{place.path}", self.objective)
    return ScoreVar(f"${place.path}", self.objective)

  @overload
  def get_cmdop(self, op: lowering.Place) -> CommandVar:
    ...

  @overload
  def get_cmdop(self, op: lowering.ImmValue) -> CommandOp:
    ...

  def get_cmdop(self, op: lowering.Operand) -> CommandOp:
    match op:
      case lowering.Place():
        match op.layout_type:
          case lowering.ScalarType.INT:
            return self.score_holder(op)

          case lowering.ScalarType.LABEL:
            return NBTVar(op.path, self.stg)

      case lowering.ImmValue(val=val):
        match op.kind:
          case lowering.ScalarType.INT:
            return ScoreImm(int(val))

          case lowering.ScalarType.LABEL:
            return NBTImm(json.dumps(op.val))

  def get_getcmd(self, var: CommandVar):
    match var:
      case ScoreVar():
        return ScoreGet(var)

      case NBTVar():
        return NBTGet(var)

  def emit_copy(self, dst: CommandOp, src: CommandOp):
    match dst, src:
      case ScoreVar(), ScoreVar() | ScoreImm():
        self.emit(Scoreboard(dst, ScoreOp.SET, src))

      case ScoreVar(), NBTImm():
        self.emit(Scoreboard(dst, ScoreOp.SET, ScoreImm(int(src.val))))

      case ScoreVar(), NBTVar():
        # self.emit(
        #   Execute((StoreScore(dst, StoreMode.RESULT),), self.get_getcmd(src))
        # )
        raise NotImplementedError  # ensured by semantic pass

      case NBTVar(), ScoreVar():
        self.emit(Execute(
          (StoreNBT(dst, StoreMode.RESULT, NBTDtype.INT, 1),),
          self.get_getcmd(src)
        ))

      case NBTVar(), NBTVar() | NBTImm():
        self.emit(DataModify(dst, src))

      case _:
        raise NotImplementedError

  def emit_binop(
    self,
    dst: CommandVar,
    lhs: CommandOp,
    bop: lowering.BinaryOp,
    rhs: CommandOp,
  ) -> None:
    match dst, lhs, rhs:
      case ScoreVar() as d, ScoreImm(val=a), ScoreImm(val=b):
        self.emit(Scoreboard(d, ScoreOp.SET, ScoreImm(_fold(bop, a, b))))

      case (
        ScoreVar() as d,
        (ScoreVar() | ScoreImm()) as l,
        (ScoreVar() | ScoreImm()) as r,
      ):
        match _classify(bop):
          case CompareOp() as cop:
            self.emit(Execute(
              (StoreScore(d, StoreMode.SUCCESS), IfScore.of(l, cop, r)), None
            ))
          case ScoreOp() as sop:
            self._emit_arith(d, l, sop, r)
        
      case (
        ScoreVar() as d,
        (NBTVar() | NBTImm()) as l,
        (NBTVar() | NBTImm()) as r,
      ) if _classify(bop) in (CompareOp.CEQ, CompareOp.CNE):
        self._emit_nbt_eq(d, l, r, ne=_classify(bop) == CompareOp.CNE)

      case _:
        raise NotImplementedError

  def _emit_nbt_eq(
    self, dst: ScoreVar, lhs: NBTVar | NBTImm, rhs: NBTVar | NBTImm, ne: bool
  ) -> None:
    if isinstance(lhs, NBTVar) and isinstance(rhs, NBTImm):
      clause = IfNBT.try_of(lhs, rhs, neg=ne)
      if clause is not None:
        self.emit(Execute((StoreScore(dst, StoreMode.SUCCESS), clause), None))
        return

    self.emit(DataRemove(self.comp_tmp))
    self.emit(DataModify(self.comp_tmp, lhs))
    self.emit(Execute(
      (StoreScore(dst, StoreMode.SUCCESS),),
      DataModify(self.comp_tmp, rhs),
    ))
    if not ne:
      self.emit(Execute((
        StoreScore(dst, StoreMode.SUCCESS),
        IfScore(dst, CompareOp.CEQ, ScoreImm(0)),
      ), None))

  def _emit_arith(
    self,
    dst: ScoreVar,
    lhs: ScoreVar | ScoreImm,
    sop: ScoreOp,
    rhs: ScoreVar | ScoreImm,
  ) -> None:
    if rhs == dst and lhs != dst:
      if sop in {ScoreOp.ADD, ScoreOp.MUL, ScoreOp.CGT, ScoreOp.CLT}:
        lhs, rhs = rhs, lhs
      else:
        self.emit_copy(self.binop_tmp, rhs)
        rhs = self.binop_tmp

    if lhs != dst:
      self.emit_copy(dst, lhs)

    match rhs:
      case ScoreImm(val=val) if sop not in {ScoreOp.ADD, ScoreOp.SUB}:
        rhs = self.const_holder(val)
      case _:
        pass

    self.emit(Scoreboard(dst, sop, rhs))

  def get_cond(self, var: CommandVar, neg: bool = False) -> ExecuteClause:
    match var:
      case ScoreVar():
        cop = (CompareOp.CNE, CompareOp.CEQ)[neg]
        return IfScore(var, cop, ScoreImm(0))

      case NBTVar():
        raise NotImplementedError  # ensured by semantic pass


  def emit_setzero(self, dst: CommandVar):
    match dst:
      case ScoreVar():
        self.emit(Scoreboard(dst, ScoreOp.SET, ScoreImm(0)))

      case NBTVar():
        self.emit(DataModify(dst, NBTImm("0")))

  def emit_call(self, func: CommandOp):
    match func:
      case ScoreVar():
        # TODO
        raise NotImplementedError

      case ScoreImm():
        # TODO
        raise NotImplementedError

      case NBTVar():
        mid = "func"
        self.emit(DataModify(self.macro_stg.get_nbtvar(mid), func))
        self.emit(Call(self.ns, MacroImm(mid)))

      case NBTImm():
        self.emit(Call(self.ns, func))

      case MacroImm():
        self.emit(Call(self.ns, func))

  def emit_jump(self, func: CommandOp, cond: ExecuteClause | None):
    self.emit_call(func)
    call = self.pop()
    cmd = ReturnCmd(call)
    if cond is not None:
      cmd = Execute((cond,), cmd)
    self.emit(cmd)


class DatapackGen:
  def __init__(self, ir_funcs: list[lowering.IRFunc], ns: str):
    self.ir_funcs = ir_funcs
    
    self.ns = ns
    self.stg = Storage(ns, "main")
    self.macro_stg = Storage(ns, "args")
    self.binop_tmp = ScoreVar("#tmp", self.ns)
    self.comp_tmp = self.stg.get_nbtvar("0c")
    self.objective = ns
    self.setup_func = ".setup"
    self.consts: set[int] = set()

    self.funcs: dict[str, MCFunction] = {}
    self.fb: MCFuncBuilder = None  # type: ignore
  
  @contextmanager
  def with_func(self, name: str):
    prev = self.fb
    self.fb = MCFuncBuilder(
      name,
      self.ns,
      self.stg,
      self.macro_stg,
      self.binop_tmp,
      self.comp_tmp,
      self.objective,
      self.setup_func,
      self.consts,
    )
    try:
      yield self.fb
    
    finally:
      self.funcs[name] = self.fb.func
      self.fb = prev
  
  def write_setup(self):
    with self.with_func(self.setup_func):
      self.fb.emit(
        PlainCommand(f"scoreboard objectives add {self.ns} dummy")
      )
      for c in sorted(self.consts):
        self.fb.emit(
          Scoreboard(self.fb.const_holder(c), ScoreOp.SET, ScoreImm(c))
        )

  def gen(self) -> dict[str, MCFunction]:
    for func in self.ir_funcs:
      optimize.optimize(func)

      for block in func.blocks:
        block.scalarize()

        block_name = block.addr.val
        with self.with_func(block_name):
          for inst in block.body:
            self.gen_inst(inst)

          if block.term is not None:
            self.gen_term(block.term)

    self.write_setup()
    return self.funcs

  def gen_inst(self, inst: lowering.IR):
    match inst:
      case lowering.Copy(dst=d, src=s):
        dst = self.fb.get_cmdop(d)
        src = self.fb.get_cmdop(s)
        self.fb.emit_copy(dst, src)

      case lowering.SetZero(dst=d):
        dst = self.fb.get_cmdop(d)
        self.fb.emit_setzero(dst)

      case lowering.BinOp(dst=d, lhs=l, rhs=r, bop=bop):
        dst = self.fb.get_cmdop(d)
        lhs = self.fb.get_cmdop(l)
        rhs = self.fb.get_cmdop(r)

        assert isinstance(dst, CommandVar)
        self.fb.emit_binop(dst, lhs, bop, rhs)

      case lowering.Call(callee=lowering.IRFunc(addr=addr)):
        mcf = self.fb.get_cmdop(addr)
        self.fb.emit_call(mcf)
      
      case lowering.IndirectCall(target=target):
        mcf = self.fb.get_cmdop(target)
        self.fb.emit_call(mcf)

  def gen_term(self, term: lowering.Terminator):
    match term:
      case lowering.Jump(target=target):
        mcf = self.fb.get_cmdop(target.addr)
        self.fb.emit_jump(mcf, None)

      case lowering.Branch(cond=c, then=t, orelse=e):
        cond = self.fb.get_cmdop(c)
        then = self.fb.get_cmdop(t.addr)
        orelse = self.fb.get_cmdop(e.addr)
        self.fb.emit_jump(then, self.fb.get_cond(cond))
        self.fb.emit_jump(orelse, None)

      case lowering.IndirectJump(target=target):
        mcf = self.fb.get_cmdop(target)
        self.fb.emit_jump(mcf, None)

      case lowering.Ret():
        self.fb.emit(ReturnVal(0))


  def gen_setzero(self, inst: lowering.SetZero):
    match inst.dst:
      case lowering.Place() as dst:
        holder = self.fb.score_holder(dst)
        self.fb.emit_setzero(holder)

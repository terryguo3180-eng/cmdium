from __future__ import annotations

from typing import TypeIs

from cmdium import lowering


def _wrap32(x: int) -> int:
  return (x + 2**31) % 2**32 - 2**31


def fold(bop: lowering.BinaryOp, a: int, b: int) -> int | None:
  match bop:
    case lowering.BinaryOp.ADD: r = a + b
    case lowering.BinaryOp.SUB: r = a - b
    case lowering.BinaryOp.MUL: r = a * b
    case lowering.BinaryOp.DIV:
      if b == 0:
        return None
      r = a // b
    case lowering.BinaryOp.MOD:
      if b == 0:
        return None
      r = a % b
    
    case lowering.BinaryOp.OR: r = max(a, b)
    case lowering.BinaryOp.AND: r = min(a, b)
    case lowering.BinaryOp.CEQ: r = int(a == b)
    case lowering.BinaryOp.CNE: r = int(a != b)
    case lowering.BinaryOp.CLT: r = int(a < b)
    case lowering.BinaryOp.CLE: r = int(a <= b)
    case lowering.BinaryOp.CGT: r = int(a > b)
    case lowering.BinaryOp.CGE: r = int(a >= b)
  
  return _wrap32(r)


def _imm(v: int) -> lowering.ImmValue:
  return lowering.ImmValue(str(v), lowering.ScalarType.INT)


def _is_int_imm(op: lowering.Operand) -> TypeIs[lowering.ImmValue]:
  return (
    isinstance(op, lowering.ImmValue) and op.kind == lowering.ScalarType.INT
  )


def prune_unreachable(func: lowering.IRFunc) -> None:
  seen: set[lowering.BasicBlock] = set()
  stack = [func.entry]
  while stack:
    b = stack.pop()
    if b in seen:
      continue
    seen.add(b)
    if b.term is not None:
      stack.extend(b.term.successors())
  func.blocks[:] = [b for b in func.blocks if b in seen]


def merge_blocks(func: lowering.IRFunc) -> None:
  while True:
    preds = {b: 0 for b in func.blocks}
    for b in func.blocks:
      assert b.term is not None
      for s in b.term.successors():
        preds[s] += 1

    for a in func.blocks:
      if not isinstance(a.term, lowering.Jump):
        continue
      b = a.term.target
      if b is a or b is func.entry or preds[b] != 1:
        continue
      a.body.extend(b.body)
      a.term = b.term
      func.blocks.remove(b)
      break
    else:
      return


def local_opt(block: lowering.BasicBlock) -> None:
  env: dict[lowering.Place, lowering.Operand] = {}

  def sub(op: lowering.Operand) -> lowering.Operand:
    return env.get(op, op) if isinstance(op, lowering.Place) else op

  def kill(p: lowering.Place) -> None:
    env.pop(p, None)
    for k in [k for k, v in env.items() if v == p]:
      del env[k]

  out: list[lowering.IR] = []
  for inst in block.body:
    match inst:
      case lowering.Copy(dst=d, src=s):
        s = sub(s)
        if s == d:  # x = x
          continue
        kill(d)
        env[d] = s
        out.append(lowering.Copy(d, s))

      case lowering.BinOp(dst=d, lhs=l, bop=bop, rhs=r):
        l, r = sub(l), sub(r)
        kill(d)
        v = None
        if _is_int_imm(l) and _is_int_imm(r):
          v = fold(bop, int(l.val), int(r.val))
        if v is not None:
          c = _imm(v)
          env[d] = c
          out.append(lowering.Copy(d, c))
        else:
          out.append(lowering.BinOp(d, l, bop, r))

      case lowering.Call() | lowering.IndirectCall():
        # a callee may write any global; only temps survive
        for k in [
          k for k, v in env.items()
          if not (k.temp and (isinstance(v, lowering.ImmValue) or v.temp))
        ]:
          del env[k]
        out.append(inst)

      case lowering.SetZero():
        raise AssertionError("run lowering.scalarize before optimizing")

  block.body = out

  match block.term:
    case lowering.Branch(cond=c, then=t, orelse=e):
      c = sub(c)
      if t is e:
        block.term = lowering.Jump(t)
      elif isinstance(c, lowering.ImmValue):
        block.term = lowering.Jump(t if int(c.val) else e)
      else:
        block.term = lowering.Branch(c, t, e)

    case _:
      pass


def remove_dead_temps(func: lowering.IRFunc) -> None:
  while True:
    used: set[lowering.Operand] = set()
    for b in func.blocks:
      for inst in b.body:
        match inst:
          case lowering.Copy(src=s):
            used.add(s)
          case lowering.BinOp(lhs=l, rhs=r):
            used.add(l)
            used.add(r)
          case lowering.IndirectCall(target=t):
            used.add(t)
          case _:
            pass

      match b.term:
        case lowering.Branch(cond=c):
          used.add(c)
        case lowering.IndirectJump(target=t):
          used.add(t)
        case _:
          pass

    changed = False
    for b in func.blocks:
      keep: list[lowering.IR] = []
      for inst in b.body:
        match inst:
          case lowering.Copy(dst=d) | lowering.BinOp(dst=d) if d.temp and d not in used:
            changed = True
          case _:
            keep.append(inst)
      b.body = keep

    if not changed:
      return


def optimize(func: lowering.IRFunc) -> None:
  while True:
    before = str(func)
    prune_unreachable(func)
    merge_blocks(func)
    for b in func.blocks:
      local_opt(b)
    remove_dead_temps(func)
    if str(func) == before:
      return

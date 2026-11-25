from __future__ import annotations

from collections.abc import Iterator

from cmdium import lowering


def fold(bop: lowering.BinaryOp, a: int, b: int) -> int | None:
  # `/=` is floor division and `%=` is the positive remainder in Minecraft
  # (verified on a 1.21.1 server), which is exactly Python's `//` and `%`.
  # For a zero divisor the command fails and leaves the target unchanged, so
  # there is nothing to fold; the front end rejects constant zero divisors.
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
    
  return (r + 2**31) % 2**32 - 2**31


def _succ_blocks(term: lowering.Terminator) -> Iterator[lowering.BasicBlock]:
  for s in term.successors():
    yield s


def prune_unreachable(func: lowering.IRFunc) -> None:
  seen: set[lowering.BasicBlock] = set()
  stack = [func.entry]
  while stack:
    b = stack.pop()
    if b in seen:
      continue
    seen.add(b)
    if b.term is not None:
      stack.extend(_succ_blocks(b.term))
  func.blocks[:] = [b for b in func.blocks if b in seen]


def merge_blocks(func: lowering.IRFunc) -> None:
  while True:
    preds = {b: 0 for b in func.blocks}
    for b in func.blocks:
      assert b.term is not None
      for s in _succ_blocks(b.term):
        preds[s] += 1

    for a in func.blocks:
      if not isinstance(a.term, lowering.Jump):
        continue
      b = a.term.target.bb
      if b is a or b is func.entry or preds[b] != 1:
        continue
      a.body.extend(b.body)
      a.term = b.term
      func.blocks.remove(b)
      break
    else:
      return


def local_opt(
  block: lowering.BasicBlock, exported: set[lowering.Var]
) -> None:
  env: dict[lowering.Var, lowering.Operand] = {}

  def sub(op: lowering.Operand) -> lowering.Operand:
    return env.get(op, op) if isinstance(op, lowering.Var) else op

  def kill(p: lowering.Var) -> None:
    env.pop(p, None)
    for k in [k for k, v in env.items() if v == p]:
      del env[k]

  def invalidate_for_call() -> None:
    for k, v in list(env.items()):
      stable = not (
        k in exported or (isinstance(v, lowering.Var) and v in exported)
      )
      if not stable:
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
        if isinstance(l, lowering.IntImm) and isinstance(r, lowering.IntImm):
          v = fold(bop, l.val, r.val)
        if v is not None:
          c = lowering.IntImm(v)
          env[d] = c
          out.append(lowering.Copy(d, c))
        else:
          out.append(lowering.BinOp(d, l, bop, r))

      case lowering.Call(callee=callee, args=args, rets=rets):
        args = tuple(sub(a) for a in args)
        invalidate_for_call()
        for r in rets:
          if r is not None:
            kill(r)
        out.append(lowering.Call(callee, args, rets))

      case lowering.IndirectCall():
        invalidate_for_call()
        out.append(inst)

  block.body = out

  match block.term:
    case lowering.Branch(cond=c, then=t, orelse=e):
      c = sub(c)
      if t == e:
        block.term = lowering.Jump(t)
      elif isinstance(c, lowering.IntImm):
        block.term = lowering.Jump(t if c.val else e)
      else:
        assert isinstance(c, lowering.Var)
        block.term = lowering.Branch(c, t, e)

    case lowering.Ret(values=values):
      block.term = lowering.Ret(tuple(sub(v) for v in values))

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
          case lowering.Call(args=args):
            used.update(args)
          case lowering.IndirectCall(target=t):
            used.add(t)

      match b.term:
        case lowering.Branch(cond=c):
          used.add(c)
        case lowering.Jump(target=t) | lowering.IndirectJump(target=t):
          used.add(t)
        case lowering.Ret(values=values):
          used.update(values)
        case _:
          pass

    changed = False
    for b in func.blocks:
      keep: list[lowering.IR] = []
      for inst in b.body:
        match inst:
          case (
            lowering.Copy(dst=d) | lowering.BinOp(dst=d)
          ) if not (d in func.exports or d in used):
            changed = True
          case lowering.Call(callee=callee, args=args, rets=rets):
            new_rets: list[lowering.Var | None] = []
            for r in rets:
              if not (r is None or r in func.exports or r in used):
                changed = True
                r = None
              new_rets.append(r)
            keep.append(lowering.Call(callee, args, tuple(new_rets)))
          case _:
            keep.append(inst)
      b.body = keep

    if not changed:
      return


def optimize(func: lowering.IRFunc) -> None:
  exported = set(func.exports)
  while True:
    before = str(func)
    prune_unreachable(func)
    merge_blocks(func)
    for b in func.blocks:
      local_opt(b, exported)
    remove_dead_temps(func)
    if str(func) == before:
      return

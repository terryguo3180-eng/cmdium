from collections.abc import Iterator
from typing import get_args

from cmdium import lowering
from cmdium import syntax


def leaves(cls: type) -> Iterator[type]:
  subs = cls.__subclasses__()
  if not subs:
    yield cls
  for s in subs:
    yield from leaves(s)


UNIONS: list[tuple[object, type]] = [
  (syntax.Declaration, syntax.DeclarationBase),
  (syntax.RawStatement, syntax.RawStatementBase),
  (syntax.Statement, syntax.StatementBase),
  (syntax.Expression, syntax.ExpressionBase),
  (syntax.TypeExpr, syntax.TypeExprBase),
  (lowering.IR, lowering.IRBase),
]

if __name__ == "__main__":
  ok = True
  for union, base in UNIONS:
    members = set(get_args(union))
    want = set(leaves(base))
    missing = sorted(c.__name__ for c in want - members)
    extra = sorted(c.__name__ for c in members - want)
    if missing or extra:
      ok = False
    status = "in sync" if not (missing or extra) else (
      f"missing from union: {missing}, not in hierarchy: {extra}"
    )
    print(f"{base.__name__:18} {status}")
  raise SystemExit(0 if ok else 1)

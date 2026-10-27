from typing import NamedTuple


class Span(NamedTuple):
  file: str
  spos: int
  epos: int


class CompileError(Exception):
  def __init__(self, msg: str, span: Span):
    super().__init__(msg)
    self.msg = msg
    self.span = span


def get_lines(
  src: str, spos: int, epos: int
) -> list[tuple[int, str, int, int]]:
  if spos >= epos:
    return []

  lines = []
  i = 0
  for ln, line in enumerate(src.split("\n"), 1):
    if i < epos and i + len(line) >= spos:
      start = max(spos - i, 0)
      end = min(epos - i, len(line))
      if start <= end:
        lines.append((ln, line, start, end))
    i += len(line) + 1

  return lines


_RESET      = "\x1b[0m"
_BOLD       = "\x1b[1m"
_GRAY       = "\x1b[90m"
_BOLD_RED   = "\x1b[1;31m"
_BOLD_BLUE  = "\x1b[1;34m"


def format_error(err: CompileError, colored: bool = False):
  file, spos, epos = err.span
  epos = max(epos, spos + 1)

  with open(file, encoding="utf-8") as f:
    src = f.read().expandtabs(1)

  ln = src.count("\n", 0, spos) + 1
  cl = spos - src.rfind("\n", 0, spos)

  def paint(text: str, code: str) -> str:
    if not colored:
      return text
    return f"{code}{text}{_RESET}"

  loc = f"{file}:{ln}:{cl}"
  fmt = f"{paint(loc, _BOLD)}: {paint('error', _BOLD_RED)}: {err.msg}\n"

  maxln = src.count("\n") + 1
  leadz = len(str(maxln))
  for ln, line, ls, le in get_lines(src, spos, epos):
    raw_prefix = f"  {str(ln).zfill(leadz)} | "
    prefix = paint(raw_prefix[:-3], _BOLD_BLUE) + paint(raw_prefix[-3:], _GRAY)

    fmt += f"{prefix}{line}\n"
    fmt += (
      " " * len(raw_prefix)
      + " " * ls
      + paint("^" * max(le - ls, 1), _BOLD_RED)
      + "\n"
    )

  return fmt.rstrip("\n")

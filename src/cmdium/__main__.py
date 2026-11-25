from __future__ import annotations

import argparse
import re
import sys
import warnings

from pathlib import Path

from cmdium import datapack
from cmdium import error
from cmdium import syntax
from cmdium import codegen
from cmdium import lowering
from cmdium import semantic


# pack_format 48 = Minecraft 1.21 (the first version using `function/`
# instead of `functions/` in data packs).
DEFAULT_PACK_FORMAT = 48
FIRST_SINGULAR_PACK_FORMAT = 48

EMIT_STAGES = ("ast", "ir", "mcfunction", "pack")


def default_namespace(path: Path) -> str:
  ns = re.sub(r"[^a-z0-9_.\-]", "_", path.stem.lower())
  return ns or "custom"


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="cmdium",
    description="Compile a .cmdium source file into a Minecraft data pack.",
  )
  p.add_argument("input", type=Path, help="source file to compile")

  out = p.add_argument_group("output")
  out.add_argument(
    "-o", "--output", type=Path, metavar="PATH",
    help="output path; a '.zip' suffix produces a zip, anything else a "
         "directory (default: ./<namespace>.zip)",
  )
  out.add_argument(
    "--format", choices=("auto", "dir", "zip"), default="auto",
    help="force the output format instead of guessing from --output "
         "(default: auto)",
  )
  out.add_argument(
    "--emit", choices=EMIT_STAGES, default="pack",
    help="stop after a stage and print its result to stdout instead of "
         "writing a data pack: ast, ir, mcfunction (default: pack)",
  )
  out.add_argument(
    "--check", action="store_true",
    help="only parse and type-check; write nothing (implies no output)",
  )

  pack = p.add_argument_group("data pack")
  pack.add_argument(
    "-n", "--namespace", metavar="NAME",
    help="data pack namespace, only a-z 0-9 _ . - allowed "
         "(default: derived from the input file name)",
  )
  pack.add_argument(
    "--pack-format", type=int, default=DEFAULT_PACK_FORMAT, metavar="N",
    help=f"pack_format in pack.mcmeta (default: {DEFAULT_PACK_FORMAT})",
  )
  pack.add_argument(
    "-d", "--description", default="", metavar="TEXT",
    help="description in pack.mcmeta (default: empty)",
  )
  pack.add_argument(
    "--entry", default=".main", metavar="NAME",
    help="name of entry function (default: '.main')",
  )
  pack.add_argument(
    "--setup", default=".setup", metavar="NAME",
    help="name of the setup function generated for you and registered in "
         "the load tag (default: '.setup')",
  )
  pack.add_argument(
    "--legacy-names", action=argparse.BooleanOptionalAction, default=None,
    help="use 'functions/' instead of 'function/' directories "
         f"(default: on when pack-format < {FIRST_SINGULAR_PACK_FORMAT})",
  )
  pack.add_argument(
    "--autorun", action="store_true",
    help="also run the program's entry function on world load, after the "
         "setup function",
  )

  diag = p.add_argument_group("diagnostics")
  diag.add_argument(
    "-W", "--werror", action="store_true",
    help="treat warnings as errors",
  )
  diag.add_argument(
    "--color", choices=("auto", "always", "never"), default="auto",
    help="colorize diagnostics (default: auto)",
  )
  diag.add_argument(
    "-q", "--quiet", action="store_true",
    help="don't print the 'wrote ...' summary",
  )
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)

  if args.color == "always":
    colored = True
  elif args.color == "never":
    colored = False
  else:
    colored = sys.stderr.isatty()

  file: Path = args.input
  try:
    src = file.read_text(encoding="utf-8")
  except OSError as e:
    print(f"cmdium: cannot read {str(file)!r}: {e.strerror}", file=sys.stderr)
    return 1

  path = str(file)

  # parse & semantic
  try:
    with warnings.catch_warnings(record=True) as warns:
      warnings.simplefilter("always")
      tree = syntax.parse(src, path)
      if args.emit == "ast":
        print(tree)
      semantic.Semantic(tree, path).analyze()

  except error.CompileError as err:
    print(error.format_error(err, colored), file=sys.stderr)
    return 1

  n_warn = 0
  for w in warns:
    if isinstance(w.message, error.CompileWarning):
      n_warn += 1
      print(error.format_error(w.message, colored), file=sys.stderr)
  if args.werror and n_warn:
    print(f"cmdium: {n_warn} warning(s) treated as errors", file=sys.stderr)
    return 1

  if args.check or args.emit == "ast":
    return 0

  # lowering
  ir_funcs = lowering.Lowering(tree, args.entry).lower()
  if args.emit == "ir":
    print("\n\n".join(str(f) for f in ir_funcs))
    return 0

  # codegen
  namespace = args.namespace or default_namespace(file)
  mc_funcs = codegen.DatapackGen(
    ir_funcs, namespace, setup=args.setup
  ).gen()

  if args.emit == "mcfunction":
    for name, fn in mc_funcs.items():
      print(f"# {name}.mcfunction")
      for cmd in "\n".join(fn.cmds).split("\n"):
        print(f"  {cmd}")
      print()
    return 0

  # write the data pack
  out: Path = args.output or Path(f"{namespace}.zip")
  legacy = args.legacy_names
  if legacy is None:
    legacy = args.pack_format < FIRST_SINGULAR_PACK_FORMAT
  load = (args.setup,)

  if args.autorun:
    load += (args.entry,)

  try:
    result = datapack.write_datapack(
      mc_funcs, out,
      namespace=namespace,
      pack_format=args.pack_format,
      description=args.description,
      load=load,
      legacy_names=legacy,
      fmt=args.format,
    )
  except ValueError as e:
    print(f"cmdium: error: {e}", file=sys.stderr)
    return 1
  except OSError as e:
    print(f"cmdium: cannot write {str(out)!r}: {e.strerror}", file=sys.stderr)
    return 1

  if not args.quiet:
    print(
      f"wrote {result} ({len(mc_funcs)} functions, namespace {namespace!r})"
    )
  return 0


if __name__ == "__main__":
  try:
    code = main()
    sys.stdout.flush()
  except BrokenPipeError:
    import os
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    code = 1
  sys.exit(code)

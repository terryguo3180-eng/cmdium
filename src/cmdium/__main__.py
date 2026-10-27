from cmdium import syntax
from cmdium.error import CompileError, format_error
from cmdium.semantic import Semantic
from cmdium.lowering import Lowering
from cmdium.codegen import DatapackGen


file = "../demos/test.cmdium"

with open(file, encoding="utf-8") as f:
  src = f.read()

try:
  tree = syntax.parse(src, file)
  typed = Semantic(tree, file).analyze()

except CompileError as err:
  print(format_error(err, True))

else:
  insts = Lowering(tree).lower()
  funcs = DatapackGen(insts, "cmdium").render()

  for func, cmds in funcs.items():
    print("# " + func + ".mcfunction")

    for cmd in cmds:
      print("  " + cmd)

    print()

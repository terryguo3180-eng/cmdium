from cmdium import syntax
from cmdium.semantic import Semantic
from cmdium.lowering import Lowering
from cmdium.codegen import DatapackGen


with open("../demos/test.cmdium") as f:
  src = f.read()

tree = syntax.parse(src)
typed = Semantic(tree).analyze()
insts = Lowering(tree).lower()
funcs = DatapackGen(insts, "cmdium").render()

for func, cmds in funcs.items():
  print("# " + func + ".mcfunction")

  for cmd in cmds:
    print("  " + cmd)

  print()

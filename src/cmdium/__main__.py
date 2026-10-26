from cmdium import syntax
from cmdium.lowering import Lowerer, DatapackRenderer


with open("../demos/test.cmdium") as f:
  src = f.read()

tree = syntax.parse(src)

lowerer = Lowerer(tree)
insts = lowerer.visit()

renderer = DatapackRenderer(insts, "cmdium")

for func, cmds in renderer.render().items():
  print("# " + func + ".mcfunction")

  for cmd in cmds:
    print("  " + cmd)

  print()

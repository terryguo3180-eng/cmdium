import warnings

from cmdium import syntax
from cmdium import error
from cmdium.codegen import DatapackGen
from cmdium.lowering import Lowering
from cmdium.optimize import optimize
from cmdium.semantic import Semantic


file = "../demos/test.cmdium"

with open(file, encoding="utf-8") as f:
  src = f.read()

print("---")
print(src)
print("---")
print()

try:
  with warnings.catch_warnings(record=True) as warns:
    tree = syntax.parse(src, file)
    print(tree)
    typed = Semantic(tree, file).analyze()

  for warn in warns:
    if isinstance(warn.message, error.CompileWarning):
      print(error.format_error(warn.message, True))

except error.CompileError as err:
  print(error.format_error(err, True))
  exit(1)

else:
    ir_funcs = Lowering(tree).lower()
    print("---")
    for ir_func in ir_funcs:
      print(str(ir_func))
      print()
    print("---")

    print("---")
    for ir_func in ir_funcs:
      optimize(ir_func)
      print(str(ir_func))
      print()
    print("---")

    mc_funcs = DatapackGen(ir_funcs, "cmdium").gen()

    print("---")
    for name, mc_func in mc_funcs.items():
      print("# " + name + ".mcfunction")

      for cmd in mc_func.cmds:
        print("  " + str(cmd))
      print()
    print("---")
    print()

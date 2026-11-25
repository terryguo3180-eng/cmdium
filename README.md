# cmdium

**cmdium** is a compiler that turns a small, statically-typed high-level
language into a Minecraft Java Edition data pack (`.mcfunction`).

> **Status:** early development. Many features are still missing (see [Roadmap](#roadmap)).

Target: **Minecraft JE 1.21+** (`pack_format` 48 by default).

---

## Features (current)

- Static typing with `i32`, `bool`, `str`, tuples, structs and function types
- Functions with parameters and return values
- Function values: assign, pass, return and call them indirectly (each target
  cell gets a macro trampoline, no candidate scan)
- Control flow: `if` / `else`, `while`, `break`, `continue`, `return`
- Arithmetic, comparison and logical operators
- Variable declarations, assignment and compound assignment (`+=`, `-=`, …)
- Struct and tuple literals + field access
- `@export` attribute (exposes a variable to the scoreboard with a custom name)
- Full pipeline: parse → type-check → lower to IR → codegen → data pack
- Can emit intermediate representations (`ast`, `ir`, `mcfunction`) for debugging
- Outputs either a directory or a `.zip` data pack

### Explicitly **not** supported yet

- Recursion
- Pointers / references
- Arrays / lists
- Raw Minecraft commands (`say`, `execute as …`, entity/block NBT, etc.)
- Macros / compile-time evaluation beyond constant folding
- Modules / multiple files

The language currently only does pure computation using scoreboards + storage.

---

## Quick start

```bash
# install (when published)
pip install cmdium          # or: pip install -e .

# compile a source file
cmdium hello.cmdium

# produces hello.zip (or a directory) containing a ready-to-use data pack
```

Place the generated data pack into your world's `datapacks/` folder and `/reload`.

By default the entry point is the top-level code (compiled into `.main`).  
A setup function named `.setup` is always generated and registered in
`#minecraft:load`; pick another name with `--setup`.

---

## Language overview

### Types

| Type          | Description                          |
|---------------|--------------------------------------|
| `i32`         | 32-bit signed integer                |
| `bool`        | Boolean (`T` / `F`)                  |
| `str`         | String                               |
| `(T, U, ...)` | Tuple                                |
| `struct`      | User-defined product type            |
| `func(...)`   | Function type                        |

### Example

```cmdium
func is_prime(n: i32) -> bool {
  if (n < 2) return F;
  if (n == 2 | n == 3) return T;
  if (n % 2 == 0) return F;

  i: i32 = 3;
  while (i <= n / i) {
    if (n % i == 0) {
      return F;
    }
    i += 1;
  }

  return T;
}

i: i32 = 0;

@export("number")
n: i32 = 0;

while (i < 100000) {
  if (is_prime(i)) {
    n += 1;
  }
  i += 1;
}
```

This program counts the number of primes below 100 000 and stores the result in the scoreboard objective under the name `$number`.

---

## Compiler usage

```text
cmdium [options] <input.cmdium>
```

### Common options

| Flag                    | Description                                      | Default              |
|-------------------------|--------------------------------------------------|----------------------|
| `-o`, `--output PATH`   | Output path (`.zip` → zip, otherwise directory)  | `./<namespace>.zip`  |
| `--format {auto,dir,zip}`| Force output format                             | `auto`               |
| `-n`, `--namespace NAME`| Data pack namespace                              | derived from filename|
| `--pack-format N`       | `pack_format` in `pack.mcmeta`                   | `48` (1.21)          |
| `-d`, `--description`   | Description written into `pack.mcmeta`           | empty                |
| `--entry NAME`          | Name of the entry function                       | `.main`              |
| `--setup NAME`          | Name of the generated setup function             | `.setup`             |
| `--autorun`             | Also run the entry function on world load        | off                  |
| `--emit {ast,ir,mcfunction,pack}`| Stop after a stage and print to stdout  | `pack`               |
| `--check`               | Only parse + type-check, write nothing           |                      |
| `-W`, `--werror`        | Treat warnings as errors                         |                      |
| `-q`, `--quiet`         | Suppress the “wrote …” message                   |                      |

### Debugging intermediate stages

```bash
cmdium --emit ast   test.cmdium   # pretty-printed AST
cmdium --emit ir    test.cmdium   # lowered IR
cmdium --emit mcfunction test.cmdium  # generated commands
```

---

## How it works (high level)

```
.cmdium source
    │
    V  syntax.py          lex + parse
   AST
    │
    V  semantic.py        name resolution + type checking
   typed AST
    │
    V  lowering.py        lower to SSA-like IR (basic blocks)
   IR functions
    │
    V  codegen.py         map IR values to global slots
   slot assignments
    │
    V  abi.py             slots → scoreboard / data / execute / function
   mcfunction commands
    │
    V  datapack.py        write pack.mcmeta + function/ + load tag
   data pack (.zip or dir)
```

Every scalar value of the program lives in one global **slot** (`slots.py`):
a numeric id plus a storage kind, either a scoreboard score or an NBT tag.
`abi.py` is the only module that knows Java Edition syntax; it names the
slots and renders the commands. There is no optimiser stage yet.

---

## Project layout

```
cmdium/
├── __main__.py      # CLI entry point
├── syntax.py        # lexer + parser + AST
├── semantic.py      # type checker & symbol table
├── lowering.py      # AST → IR
├── slots.py         # global slots: scoreboard or NBT
├── codegen.py       # IR → slot-level commands
├── abi.py           # slots & commands → mcfunction syntax
├── datapack.py      # pack writer
├── error.py         # diagnostics
└── utils.py         # small helpers
```

---

## Roadmap

Planned (rough order):

- [ ] Recursion / call stack
- [ ] Pointers / references
- [ ] Arrays / dynamic collections
- [ ] Ability to emit raw Minecraft commands / interact with entities & blocks
- [ ] Better standard library (math, string helpers, …)
- [ ] Multi-file projects / modules
- [ ] Source maps / better diagnostics
- [ ] More aggressive optimisations

Contributions and ideas are welcome once the core stabilises.

---

## Performance

Command costs were measured on a real server, in-chain with `/stopwatch`:
a `function` call is ~90 ns of overhead, a scoreboard operation ~100 ns,
a macro dispatch ~260 ns on a cache hit and ~700 ns on a miss, and a deep call
chain costs the same per level at depth 200 and depth 20 000.

See [docs/performance.md](docs/performance.md) for the tables, the measurement
method, and the codegen decisions they justify.

---

## License

MIT

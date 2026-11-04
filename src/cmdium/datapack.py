from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
  from cmdium.codegen import MCFunction


def build_files(
  funcs: Mapping[str, MCFunction],
  namespace: str,
  pack_format: int,
  description: str = "",
  load: Sequence[str] = (".setup",),
  legacy_names: bool = False,
) -> dict[str, str]:
  
  if not re.fullmatch(r"[a-z0-9_.\-]+", namespace):
    raise ValueError(
      f"invalid namespace {namespace!r}: only a-z 0-9 _ . - are allowed"
    )

  bad = [
    n for n in funcs
    if not re.fullmatch(re.compile(r"[a-z0-9_.\-/]+"), n)
    or any(seg in ("", ".", "..") for seg in n.split("/"))
  ]
  if bad:
    shown = ", ".join(repr(n) for n in sorted(bad)[:8])
    more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
    raise ValueError(
      f"function name(s) not valid in a data pack: {shown}{more}. "
      "Only a-z 0-9 _ . - / are allowed (no uppercase, no empty or '..' "
      "path segments)."
    )

  missing = [n for n in load if n not in funcs]
  if missing:
    raise ValueError(f"`load` refers to unknown function(s): {missing}")

  fdir = "functions" if legacy_names else "function"

  files: dict[str, str] = {
    "pack.mcmeta": json.dumps(
      {"pack": {"pack_format": pack_format, "description": description}},
      indent=2, ensure_ascii=False,
    ) + "\n",
  }

  for name, fn in funcs.items():
    files[f"data/{namespace}/{fdir}/{name}.mcfunction"] = "".join(
      f"{cmd}\n" for cmd in fn.cmds
    )

  if load:
    files[f"data/minecraft/tags/{fdir}/load.json"] = json.dumps(
      {"values": [f"{namespace}:{n}" for n in load]}, indent=2,
    ) + "\n"

  return files

def write_datapack(
  funcs: Mapping[str, MCFunction],
  out: str | Path,
  namespace: str,
  pack_format: int,
  description: str = "",
  load: Sequence[str] = (".setup",),
  legacy_names: bool = False,
  fmt: Literal["auto", "dir", "zip"] = "auto",
) -> Path:
  out = Path(out)
  if fmt == "auto":
    fmt = "zip" if out.suffix.lower() == ".zip" else "dir"

  files = build_files(
    funcs, namespace=namespace, pack_format=pack_format,
    description=description, load=load, legacy_names=legacy_names,
  )

  if fmt == "zip":
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
      for rel in sorted(files):
        # for reproducibility
        info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, files[rel].encode("utf-8"))
  else:
    fdir = "functions" if legacy_names else "function"
    stale = out / "data" / namespace / fdir
    if stale.is_dir():
      shutil.rmtree(stale)
  
    for rel, text in files.items():
      path = out / rel
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(text.encode("utf-8"))

  return out

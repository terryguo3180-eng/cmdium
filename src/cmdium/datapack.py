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

_NAMESPACE_RE = re.compile(r"[a-z0-9_.\-]+")
_PATH_RE = re.compile(r"[a-z0-9_.\-/]+")


def _check_namespace(namespace: str) -> None:
  if not _NAMESPACE_RE.fullmatch(namespace):
    raise ValueError(
      f"invalid namespace {namespace!r}: only a-z 0-9 _ . - are allowed"
    )


def _check_function_names(names: Sequence[str]) -> None:
  bad = [
    n for n in names
    if not _PATH_RE.fullmatch(n)
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


def _function_text(fn: MCFunction) -> str:
  return "".join(f"{cmd}\n" for cmd in fn.cmds)


def build_files(
  funcs: Mapping[str, MCFunction],
  *,
  namespace: str,
  pack_format: int,
  description: str = "",
  load: Sequence[str] = (".setup",),
  legacy_names: bool = False,
) -> dict[str, str]:
  _check_namespace(namespace)
  _check_function_names(list(funcs))

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
    files[f"data/{namespace}/{fdir}/{name}.mcfunction"] = _function_text(fn)

  if load:
    files[f"data/minecraft/tags/{fdir}/load.json"] = json.dumps(
      {"values": [f"{namespace}:{n}" for n in load]}, indent=2,
    ) + "\n"

  return files


def _write_dir(
  files: Mapping[str, str], out: Path, namespace: str, legacy_names: bool,
) -> None:
  fdir = "functions" if legacy_names else "function"
  stale = out / "data" / namespace / fdir
  if stale.is_dir():
    shutil.rmtree(stale)

  for rel, text in files.items():
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _write_zip(files: Mapping[str, str], out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  with zipfile.ZipFile(out, "w") as zf:
    for rel in sorted(files):
      info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_DEFLATED
      info.external_attr = 0o644 << 16
      zf.writestr(info, files[rel].encode("utf-8"))


def write_datapack(
  funcs: Mapping[str, MCFunction],
  out: str | Path,
  *,
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
    _write_zip(files, out)
  else:
    _write_dir(files, out, namespace, legacy_names)

  return out

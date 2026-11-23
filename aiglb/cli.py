"""aiglb command line interface.

  aiglb build script.py [-o out.glb]   run a model script (it should call scene.save())
  aiglb check model.glb                inspect a .glb (sizes, parts, textures, warnings)
  aiglb presets                        list built-in material presets and textures
  aiglb preview model.glb [-o out.png] render a quick preview image (no GPU needed)
  aiglb demo [-o demo.glb]             build & save the built-in demo rocket
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

from . import __version__, list_presets, list_textures


def _cmd_build(args: argparse.Namespace) -> int:
    if args.output:
        os.environ["AIGLB_OUTPUT"] = args.output
    if not os.path.isfile(args.script):
        print(f"error: no such file: {args.script}", file=sys.stderr)
        return 2
    runpy.run_path(args.script, run_name="__main__")
    if args.output and not os.path.exists(args.output):
        print(f"⚠ script finished but {args.output} was not created — did it call scene.save()?")
        return 1
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from .validate import format_report
    print(format_report(args.model))
    if not args.no_assembly:
        from .assembly import assembly_report_path
        print(assembly_report_path(args.model))
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    from .library import list_library, library_categories
    print("Material presets (use the name as `material=` in scene.add, or '#hexcolor'):")
    for name in list_presets():
        print(f"  {name}")
    print("\nBundled CC0 texture sets — use as \'lib:<name>\', e.g. material='lib:ambientcg/PaintedMetal004'")
    for category, count in library_categories().items():
        print(f"  {category:18s} {count:4d} sets")
    print("\n(aiglb presets --library) lists every bundled set by name")
    print("\nBuilt-in procedural textures (for texture_material(...)):")
    for name in list_textures():
        print(f"  {name}")
    return 0


def _cmd_library(args: argparse.Namespace) -> int:
    from .library import list_library
    names = list_library(category=args.category)
    header = f"Bundled CC0 texture sets ({len(names)})"
    if args.category:
        header += f" in category '{args.category}'"
    print(header + " — use as material='lib:<name>':")
    for name in names:
        print(f"  lib:{name}")
    print("\nAll CC0-1.0, free for commercial use. See aiglb/assets/TEXTURE_LICENSES.txt")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    from .preview import render_preview
    out = render_preview(args.model, out=args.output, view=args.view, size=args.size)
    print(f"✓ preview written to {out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from ._demo import build_demo_scene
    scene = build_demo_scene()
    out = args.output or os.environ.get("AIGLB_OUTPUT") or "demo_rocket.glb"
    scene.save(out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aiglb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"aiglb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="run a model script (should call scene.save())")
    p.add_argument("script")
    p.add_argument("-o", "--output", default=None, help="default output .glb (via AIGLB_OUTPUT)")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("check", help="inspect a .glb file (structure + assembly gaps)")
    p.add_argument("model")
    p.add_argument("--no-assembly", action="store_true",
                   help="skip the floating/hidden part analysis")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("presets", help="list built-in material presets and textures")
    p.set_defaults(func=_cmd_presets)

    p = sub.add_parser("library", help="list bundled CC0 texture sets ('lib:<name>')")
    p.add_argument("--category", default=None, help="filter by category (see 'aiglb presets')")
    p.set_defaults(func=_cmd_library)

    p = sub.add_parser("preview", help="render a quick preview image")
    p.add_argument("model")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--view", default="iso", choices=["iso", "hero", "front", "back", "side", "top"])
    p.add_argument("--size", type=int, default=1100)
    p.set_defaults(func=_cmd_preview)

    p = sub.add_parser("demo", help="build & save the built-in demo rocket")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

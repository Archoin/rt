"""Phase 1 smoke test 1 — render the Cornell box with the LLVM AD variant.

Run from the repo root:
    PYTHONPATH=src python scripts/smoke_test_cornell.py

The LLVM library path is configured automatically by diffrt.variants.
"""

import pathlib

from diffrt.variants import set_variant

mi = set_variant()
print("variants:", mi.variants())
print("active:  ", mi.variant())

scene = mi.load_dict(mi.cornell_box())
img = mi.render(scene, spp=16)

out_dir = pathlib.Path(__file__).resolve().parent.parent / "runs"
out_dir.mkdir(exist_ok=True)
exr_path = out_dir / "cornell.exr"
png_path = out_dir / "cornell.png"

mi.util.write_bitmap(str(exr_path), img)
mi.util.write_bitmap(str(png_path), img)
print(f"wrote {exr_path}")
print(f"wrote {png_path}")

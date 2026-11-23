"""Built-in demo scene: a cartoon rocket on a grass pad (used by `aiglb demo`)."""
from __future__ import annotations

import math

import aiglb


def build_demo_scene() -> aiglb.Scene:
    s = aiglb.Scene("aiglb_demo_rocket")

    # ground pad
    s.add("ground", aiglb.cylinder(2.1, 0.16, 48), "grass", position=(0, 0.08, 0))
    s.add("pad", aiglb.cylinder(1.1, 0.12, 32), "concrete", position=(0, 0.2, 0))

    # engine nozzle + warm glow, mounted on the pad
    s.add("nozzle", aiglb.cylinder(0.3, 0.42, 32, radius_top=0.44), "gunmetal", position=(0, 0.48, 0))

    # body: white capsule + red nose cone + gold ring seam
    s.add("body", aiglb.capsule(0.62, 1.35, 40), "porcelain", position=(0, 1.6, 0))
    s.add("nose", aiglb.cone(0.6, 0.95, 40), "plastic_red", position=(0, 2.73, 0))
    s.add("seam", aiglb.torus(0.62, 0.045, 48, 12), "gold", position=(0, 1.6, 0))

    # porthole: brass rim + glass (front, +Z)
    s.add("porthole_rim", aiglb.torus(0.26, 0.07, 40, 12), "brass",
          position=(0, 1.9, 0.60), rotation=(90, 0, 0))
    s.add("porthole_glass", aiglb.cylinder(0.23, 0.1, 32), "glass",
          position=(0, 1.9, 0.60), rotation=(90, 0, 0))

    # fins: three red wedges flared outward around the base
    fin = aiglb.wedge(0.36, 0.85, 0.55)
    for i, ang in enumerate((0, 120, 240)):
        a = math.radians(ang)
        s.add(f"fin_{i}", fin, "plastic_red",
              position=(0.55 * math.cos(a), 0.27, 0.55 * math.sin(a)),
              rotation=(0, 90 - ang, 0))

    # antenna + beacon
    s.add("antenna", aiglb.cylinder(0.022, 0.5, 8), "steel", position=(0, 3.44, 0))
    s.add("beacon", aiglb.sphere(0.06, 16, 10), "neon_pink", position=(0, 3.72, 0))
    return s

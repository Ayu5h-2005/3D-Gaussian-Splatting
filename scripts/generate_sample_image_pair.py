"""
generate_sample_image_pair.py - Generates sample baseline & updated images for 2D image change detection demo.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
out_dir = ROOT / "results" / "image_demo"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Base image (512x512 with geometric shapes, gradients, and textures)
w, h = 512, 512
img_base = Image.new("RGB", (w, h), color=(40, 45, 60))
draw = ImageDraw.Draw(img_base)

# Background gradient
for y in range(h):
    r = int(30 + (y / h) * 40)
    g = int(35 + (y / h) * 50)
    b = int(50 + (y / h) * 60)
    draw.line([(0, y), (w, y)], fill=(r, g, b))

# Architectural elements
draw.rectangle([(60, 100), (220, 420)], fill=(70, 85, 120), outline=(120, 140, 180), width=3)
draw.rectangle([(280, 180), (450, 420)], fill=(85, 100, 140), outline=(140, 160, 200), width=3)
draw.ellipse([(140, 40), (240, 140)], fill=(220, 160, 60), outline=(255, 200, 100), width=2)
draw.polygon([(100, 440), (260, 440), (180, 360)], fill=(60, 140, 100))

# Fine texture pattern for feature matching (ORB keypoints)
rng = np.random.RandomState(42)
base_np = np.array(img_base)
noise_spots = rng.randint(0, 512, (200, 2))
for x, y in noise_spots:
    base_np[max(0, y-2):min(512, y+2), max(0, x-2):min(512, x+2)] = rng.randint(180, 255, (min(512, y+2)-max(0, y-2), min(512, x+2)-max(0, x-2), 3))

path_base = out_dir / "sample_original.png"
Image.fromarray(base_np).save(path_base)
print(f"Saved sample original image: {path_base}")

# 2. Updated image: Introduce localized change (a red circular sensor pod added + translated shape)
upd_np = base_np.copy()
# Localized change 1: Bright red-orange sphere at (340, 240)
yy, xx = np.ogrid[:512, :512]
circle_mask = ((xx - 340) ** 2 + (yy - 240) ** 2) <= (35 ** 2)
upd_np[circle_mask] = [235, 65, 50]

# Localized change 2: Blue badge at (140, 300)
badge_mask = ((xx - 140) ** 2 + (yy - 300) ** 2) <= (20 ** 2)
upd_np[badge_mask] = [40, 140, 245]

path_upd = out_dir / "sample_updated.png"
Image.fromarray(upd_np).save(path_upd)
print(f"Saved sample updated image: {path_upd}")

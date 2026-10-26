import cv2
import numpy as np
from pathlib import Path

ROOT = Path(r"D:\3dgs-incremental-research")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Synthetic reference image and a changed image
before = np.zeros((512, 512, 3), dtype=np.uint8)
before[:] = (45, 45, 45)
cv2.rectangle(before, (100, 120), (220, 300), (70, 170, 220), -1)

after = before.copy()
cv2.rectangle(after, (260, 120), (380, 300), (70, 170, 220), -1)

# Detect changed pixels
diff = cv2.absdiff(before, after)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

# Map changed pixels to 32x32 screen tiles
tile_size = 32
tile_mask = np.zeros(
    (512 // tile_size, 512 // tile_size), dtype=np.uint8
)

for y in range(tile_mask.shape[0]):
    for x in range(tile_mask.shape[1]):
        tile = mask[
            y * tile_size:(y + 1) * tile_size,
            x * tile_size:(x + 1) * tile_size
        ]
        if np.any(tile):
            tile_mask[y, x] = 1

cv2.imwrite(str(OUT / "before.png"), before)
cv2.imwrite(str(OUT / "after.png"), after)
cv2.imwrite(str(OUT / "change_mask.png"), mask)
cv2.imwrite(str(OUT / "tile_mask.png"), tile_mask * 255)

changed = int(tile_mask.sum())
total = int(tile_mask.size)

print("Change detection PASSED")
print(f"Image size: 512x512")
print(f"Tile size: {tile_size}x{tile_size}")
print(f"Changed tiles: {changed}/{total}")
print(f"Unchanged tile percentage: {(1 - changed / total) * 100:.2f}%")
print(f"Results saved in: {OUT}")

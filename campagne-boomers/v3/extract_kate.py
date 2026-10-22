"""Cut Kate + her Instagram frame out of the original poster, repairing the areas the old texts covered."""
import cv2, numpy as np, pathlib
H = pathlib.Path(__file__).parent
im = cv2.imread(str(H.parent / 'v2/base/kate.png'))
raw = cv2.resize(cv2.imread('/root/.claude/uploads/d49658d4-1130-5820-9cc2-5279dd72258d/5d7f9ca8-image.png'), (2000, 1500), interpolation=cv2.INTER_AREA)
bg = np.array([239, 225, 194], np.float32)
ys, xs = np.mgrid[0:1500, 0:2000]
left_edge = 226 + (ys - 400) * 0.075
top_edge = 394 + (xs - 240) * 0.0423

# 1. repair the frame's top band (old headline was inpainted with sky blue there)
top = np.array([(224, 418), (236, 399), (252, 394), (940, 424), (958, 430), (964, 446), (964, 470), (224, 470)], np.int32)
band = np.zeros((1500, 2000), np.uint8); cv2.fillPoly(band, [top], 255)
blue = np.linalg.norm(im.astype(np.float32) - bg, axis=2) < 45
frame_col = np.median(im[440:470, 420:900].reshape(-1, 3), axis=0)
# whole header strip (avatar + name bars) is repainted flat; the poster redraws them as vector
bottom_edge = 522 + (xs - 240) * 0.0423
strip = (ys > 380) & (ys < bottom_edge) & (xs > 226) & (xs < 975)
shade = np.linspace(1.03, 0.98, 1500)[:, None, None]
flat = np.clip(frame_col[None, None, :] * shade, 0, 255)
im = np.where(strip[..., None], np.broadcast_to(flat, im.shape), im).astype(np.uint8)

# 2. foreground mask
d = np.linalg.norm(im.astype(np.float32) - bg, axis=2)
m = (d > 20).astype(np.uint8) * 255
m[:, 1430:] = 0; m[880:, 1065:] = 0; m[:385, :] = 0
m[xs < left_edge] = 0
m[(ys < top_edge) & (xs > 236)] = 0
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
keep = np.zeros_like(m)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 20000: keep[lab == i] = 255
cv2.fillPoly(keep, [top], 255)
keep[(ys < top_edge) & (xs > 250)] = 0
keep[xs < left_edge] = 0

# 3. remove old footer text (logo tail, footnote, tagline) printed over the frame
g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
txt = ((g < 75) & (ys > 1280)).astype(np.uint8) * 255
txt[1280:1365, 340:610] = 0; txt[1380:1425, 345:545] = 0; txt[1240:1320, 930:1010] = 0
txt = cv2.dilate(txt, np.ones((5, 5), np.uint8), 2)
txt[(ys > 1280) & (xs < 345) & (g < 160)] = 255
txt[keep == 0] = 0
im = cv2.inpaint(im, txt, 9, cv2.INPAINT_TELEA)

alpha = cv2.GaussianBlur(keep, (3, 3), 0)
out = np.dstack([im, alpha])
y0, y1 = np.where(keep.any(1))[0][[0, -1]]; x0, x1 = np.where(keep.any(0))[0][[0, -1]]
cv2.imwrite(str(H / 'layers/kate.png'), out[y0:y1 + 1, x0:x1 + 1])
print('bbox', x0, y0, x1, y1)

"""Cut Sophie and James out of the original posters (old footer texts removed first)."""
import cv2, numpy as np, pathlib
from PIL import Image
from rembg import remove, new_session
H = pathlib.Path(__file__).parent
UP = '/root/.claude/uploads/d49658d4-1130-5820-9cc2-5279dd72258d/'
SRC = {'sophie': ('4b6202f8-image.png', (1448, 423, 1494, 469)), 'james': ('e9bec32e-image.png', (1488, 431, 1534, 477))}
sess = new_session('isnet-general-use')
for name, (f, av) in SRC.items():
    raw = cv2.imread(UP + f); s = raw.shape[1] / 2000
    cv2.imwrite(str(H / f'layers/{name}_avatar.png'), raw[int(av[1]*s):int(av[3]*s), int(av[0]*s):int(av[2]*s)])
    im = cv2.imread(str(H.parent / f'v2/base/{name}.png'))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ys, xs = np.mgrid[0:1500, 0:2000]
    bright = cv2.dilate(g, np.ones((11, 11), np.uint8))
    txt = (g < 90) & (bright > 150) & (ys > 1270) & ((xs < 380) | (xs > 830))
    txt |= (g < 90) & (bright > 150) & (ys > 1080) & (xs > 1040)          # handwritten note + arrow
    m = cv2.dilate(txt.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), 2)
    im = cv2.inpaint(im, m, 9, cv2.INPAINT_TELEA)
    crop = im[300:1500, 0:1400]
    cut = np.array(remove(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), session=sess))
    a = cut[..., 3]
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8))
    big = np.argmax(st[1:, cv2.CC_STAT_AREA]) + 1
    keepmask = cv2.dilate((lab == big).astype(np.uint8), np.ones((15, 15), np.uint8))
    cut[..., 3] = a * keepmask
    ys2, xs2 = np.where(cut[..., 3] > 10)
    x0, x1, y0, y1 = xs2.min(), xs2.max(), ys2.min(), ys2.max()
    Image.fromarray(cut[y0:y1 + 1, x0:x1 + 1]).save(H / f'layers/{name}.png')
    print(name, 'bbox in poster', x0, y0 + 300, x1, y1 + 300)
    # background colour samples (right edge, beyond the phone)
    print(name, 'bg top/bottom BGR', np.median(im[20:80, 1920:1990].reshape(-1, 3), 0), np.median(im[1200:1260, 1920:1990].reshape(-1, 3), 0))

"""Kate 'Métro, boulot worker, dodo' reworked: bigger worker, bigger chat + QR, less empty space. 3 layouts."""
import importlib.util, pathlib, sys
from PIL import Image
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load('v3build', H.parent / 'v3/build.py')
from common import LOGO, QR_SVG, WA_ICON, wave, play

b64, RED, GOCHI = v3.b64, v3.RED, v3.GOCHI
LAYER = v3.H / 'layers/kate.png'
LW, LH = Image.open(LAYER).size
AV = b64(v3.H / 'layers/kate_avatar.png')

LAYOUTS = {
    # chat top-right, QR under the dab hand
    'A': dict(s=.98, left=510, top=625, chat=(1130, 250, 32), qr=(1480, 790, 470), scan='left'),
    # QR under the headline on the left, chat under the arm on the right
    'B': dict(s=1.04, left=560, top=560, chat=(1470, 880, 28), qr=(96, 740, 400), scan='below'),
    # biggest worker, frame bleeding off the bottom, chat + QR stacked on the right
    'C': dict(s=1.12, left=510, top=625, chat=(1440, 230, 30), qr=(1510, 800, 440), scan='left'),
}


def page(name):
    L = LAYOUTS[name]
    s = L['s']
    cx, cy, fs = L['chat']
    qx, qy, qs = L['qr']
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Métro worker dodo</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:#C2E1EF; overflow:hidden; }}
.who {{ position:absolute; left:{L['left']}px; top:{L['top']}px; width:{LW*s}px; height:{LH*s}px; z-index:1; }}
.who img {{ width:100%; height:100%; display:block; }} .who svg {{ position:absolute; inset:0; width:100%; height:100%; }}
h1 {{ position:absolute; left:86px; top:56px; font-size:160px; font-weight:500; letter-spacing:-4.5px; line-height:.98; z-index:3; }}
h1 b {{ font-weight:800; }} h1 sup {{ font-size:.45em; font-weight:500; vertical-align:.42em; margin-left:2px; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note-red {{ position:absolute; left:calc(100% + 28px); top:26%; font-family:Gochi; font-size:54px; font-weight:400; letter-spacing:0; color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
.chat {{ position:absolute; left:{cx}px; top:{cy}px; font-family:Inter; z-index:4; font-size:{fs}px; }}
.row {{ display:flex; align-items:flex-end; gap:.4em; margin-bottom:.55em; }} .row.out {{ justify-content:flex-end; }}
.av {{ width:2.2em; height:2.2em; border-radius:50%; flex:none; background:url(data:image/png;base64,{AV}) center/cover; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bub {{ background:#fff; border-radius:1em 1em 1em .3em; padding:.55em .85em; line-height:1.3; color:#111; white-space:nowrap; box-shadow:0 10px 26px rgba(20,40,90,.18); }}
.voice {{ display:flex; align-items:center; gap:.5em; height:2.9em; padding:0 1em; border-radius:1.1em 1.1em .3em 1.1em; background:#1FBE5A; color:#fff; box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.voice .wave {{ height:1.1em; gap:.12em; display:flex; align-items:center; }} .voice .wave i {{ display:block; width:.14em; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:.85em; }}
.voice svg {{ width:.7em; height:.8em; }}
.qr {{ position:absolute; left:{qx}px; top:{qy}px; width:{qs}px; height:{qs}px; background:#fff; border-radius:{qs*.09}px; box-shadow:0 18px 44px rgba(20,40,90,.24);
       display:flex; align-items:center; justify-content:center; z-index:4; }}
.qr svg {{ width:{qs*.87}px; height:{qs*.87}px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; font-family:Gochi; font-size:42px; line-height:1.0; text-align:center; transform:rotate(-8deg); white-space:nowrap; z-index:4; }}
.scan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:34px; top:1290px; height:108px; z-index:5; }}
.foot {{ position:absolute; left:34px; top:1418px; font-size:32px; color:#111; z-index:5; line-height:1.25; }} .foot small {{ font-size:23px; }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; z-index:5; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; z-index:5; }}
</style></head><body><div class="p">
<div class="who"><img src="data:image/png;base64,{b64(LAYER)}"><svg viewBox="0 0 {LW} {LH}">{v3.KATE_HEADER}</svg></div>
<h1>Métro,<br><span class="old">boulot{v3.STRIKE}<span class="note-red">← méthode boomer</span></span><br><b>worker<sup>*</sup>,</b><br>dodo.</h1>
<div class="chat">
  <div class="row"><div class="av"></div><div class="bub">Salut ! Moi c’est Kate 👋<br>Je gère tes réseaux !</div></div>
  <div class="row out"><div class="voice">{play('#fff')}<span style="width:.35em;height:.35em;border-radius:50%;background:#fff"></span>{wave('rgba(255,255,255,.9)', 24)}<span class="t">0:32</span></div></div>
  <div class="row"><div class="av"></div><div class="bub">C’est noté ! Ton post<br>part demain à 9h 🚀</div></div>
</div>
<div class="qr">{QR_SVG}</div>
<div class="scan" id="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp</div>
<svg id="arrow" style="position:absolute;z-index:4;overflow:visible" width="160" height="130" viewBox="0 0 160 130"><path id="ap" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/><path id="ah" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>
<img class="logo" src="{LOGO}" alt="Delos">
<div class="foot">*worker = employé<br><small>**boomer = un état d'esprit, pas une date de naissance</small></div>
<div class="url">www.delos.so</div><div class="tag">Vos employés IA autonomes sur mesure.</div>
</div><script>
const q=document.querySelector('.qr').getBoundingClientRect(), sc=document.getElementById('scan'), ar=document.getElementById('arrow');
const side='{L['scan']}';
if (side==='left') {{
  sc.style.left=(q.left-sc.offsetWidth-40)+'px'; sc.style.top=(q.bottom-sc.offsetHeight-20)+'px';
  ar.style.left=(q.left-170)+'px'; ar.style.top=(q.bottom-sc.offsetHeight-160)+'px';
  document.getElementById('ap').setAttribute('d','M14 124 C 26 64 70 38 148 44');
  document.getElementById('ah').setAttribute('d','M114 14 L152 44 L112 74');
}} else if (side==='below') {{
  sc.style.left=(q.left+70)+'px'; sc.style.top=(q.bottom+16)+'px'; sc.style.fontSize='36px';
  ar.style.left=(q.left-6)+'px'; ar.style.top=(q.bottom-30)+'px';
  document.getElementById('ap').setAttribute('d','M70 120 C 30 110 22 70 30 18');
  document.getElementById('ah').setAttribute('d','M8 44 L30 12 L54 42');
}} else {{
  sc.style.left=(q.right+40)+'px'; sc.style.top=(q.top+q.height*0.42)+'px';
  ar.style.left=(q.right+10)+'px'; ar.style.top=(q.top+q.height*0.42-130)+'px';
  document.getElementById('ap').setAttribute('d','M150 124 C 140 64 90 38 20 44');
  document.getElementById('ah').setAttribute('d','M48 14 L10 44 L50 74');
}}
</script></body></html>"""


if __name__ == '__main__':
    for f in H.glob('*.html'):
        f.unlink()
    for n in LAYOUTS:
        (H / f'kate-metro-{n}.html').write_text(page(n))
    print('ok')

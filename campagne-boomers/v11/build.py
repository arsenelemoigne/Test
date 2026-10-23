"""Kate poster: vertical headline on the left, with or without the phone mock-up."""
import importlib.util, pathlib, sys
from PIL import Image
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load('v3build', H.parent / 'v3/build.py')
from common import LOGO, QR_SVG, WA_ICON, wave, play

b64, RED, GOCHI, STRIKE = v3.b64, v3.RED, v3.GOCHI, v3.STRIKE

HEADS = {
    'line': lambda old: f'Métro, <span class="old">{old}{STRIKE}</span> <b>worker<sup>*</sup></b>, dodo.',
    'vertical': lambda old: f'Métro,<br><span class="old">{old}{STRIKE}</span><br><b>worker<sup>*</sup></b>,<br>dodo.',
    'dash': lambda old: f'Métro – <span class="old">{old}{STRIKE}</span><br><b>worker<sup>*</sup></b> –<br>dodo.',
}

# who: layer scale/left/top · phone: left/top/width/height/rotation · note: position next to the phone
LAYOUTS = {
    'A-vertical-telephone': dict(w='kate', bg='#C2E1EF', head='vertical', fs=172, lh=1.24, mode='phone',
        who=(.76, 720, 330), phone=(1500, 230, 450, 1070, 0), note=(0, 0)),
    'B-vertical-bulles': dict(w='kate', bg='#C2E1EF', head='vertical', fs=172, lh=1.24, mode='bubbles',
        who=(.78, 720, 330), chat=(460, 25), qr=(1580, 780, 350)),
    'C-ligne-bulles': dict(w='kate', bg='#C2E1EF', head='line', fs=118, lh=1.0, mode='bubbles',
        who=(.97, 30, 232), chat=(290, 30), qr=(1480, 680, 470)),
    'D-vertical-bulles-jaune': dict(w='kate', bg='#F6DC8E', head='vertical', fs=172, lh=1.24, mode='bubbles',
        who=(.78, 720, 330), chat=(460, 25), qr=(1580, 780, 350)),
}


def page(name):
    L = LAYOUTS[name]
    W = v3.WORKERS[L['w']]
    layer = v3.H / f"layers/{L['w']}.png"
    lw, lh = Image.open(layer).size
    s, wl, wt = L['who']
    px, py, pw, ph, rot = L.get('phone', (0, 0, 400, 800, 0))
    av = b64(v3.H / f"layers/{L['w']}_avatar.png")
    m1, m2 = W['msgs']
    nx, ny = L.get('note', (0, 0))
    qs = pw - 110 if pw < 500 else pw - 170
    cy, cfs = L.get('chat', (0, 25))
    bq = L.get('qr', (0, 0, 400))
    voice = (f'<div class="row out"><div class="voice">{play("#fff")}<span style="width:9px;height:9px;border-radius:50%;background:#fff"></span>'
             f'{wave("rgba(255,255,255,.9)", 22)}<span class="t">0:32</span></div></div>')
    rows = (f'<div class="row"><div class="av"></div><div class="bub">{m1}</div></div>{voice}'
            f'<div class="row"><div class="av"></div><div class="bub">{m2}</div></div>')
    arrow = ('<svg class="arr" width="160" height="120" viewBox="0 0 160 120"><path d="M10 110 C 30 50 80 30 146 36" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/>'
             '<path d="M112 8 L150 36 L110 66" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    scan_text = f'Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp'
    phone_html = f"""<div class="phone"><div class="screen">
  <div class="island"></div>
  <div class="status"><span>9:41</span><span>5G ▮</span></div>
  <div class="head"><svg width="16" height="28" viewBox="0 0 16 28"><path d="M13 3 L4 14 L13 25" stroke="#111" stroke-width="3.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
    <div class="av"></div><div class="name">Worker {W['name']}<small>en ligne</small></div></div>
  <div class="chat">{rows}</div>
  <div class="qrwrap"><div class="qrlabel">Scanne-moi pour me parler 👇</div><div class="qr">{QR_SVG}</div></div>
  <div class="input"><span class="plus">+</span><div class="field">Message…</div></div>
  <div class="home"></div>
</div></div>"""
    scan_html = '' if L.get('note', (0, 0)) == (0, 0) else f'<div class="scan">{scan_text}{arrow}</div>'
    bubbles = f"""<div class="bchat" style="top:{cy}px;font-size:{cfs}px">{rows}</div>
<div class="bqr" style="left:{bq[0]}px;top:{bq[1]}px;width:{bq[2]}px;height:{bq[2]}px">{QR_SVG}</div>
<div class="bscan" id="bscan">{scan_text}</div>
<svg id="barr" class="barr" width="150" height="130" viewBox="0 0 150 130"><path d="M120 120 C 40 110 20 70 26 18" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/><path d="M4 42 L26 12 L50 40" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>
<script>
const q=document.querySelector('.bqr').getBoundingClientRect(), sc=document.getElementById('bscan'), ar=document.getElementById('barr');
if (q.bottom + sc.offsetHeight + 40 < 1325) {{ sc.style.left=(q.left+q.width/2-sc.offsetWidth/2+40)+'px'; sc.style.top=(q.bottom+20)+'px'; ar.style.left=(q.left+q.width/2-sc.offsetWidth/2-90)+'px'; ar.style.top=(q.bottom-20)+'px'; }}
else {{ sc.style.left=(q.left-sc.offsetWidth-60)+'px'; sc.style.top=(q.bottom-sc.offsetHeight-40)+'px'; ar.style.left=(q.left-150)+'px'; ar.style.top=(q.bottom-sc.offsetHeight-170)+'px'; ar.style.transform='scaleX(-1) rotate(80deg)'; }}
</script>"""
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Métro worker dodo</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:{L['bg']}; overflow:hidden; }}
.who {{ position:absolute; left:{wl}px; top:{wt}px; width:{lw*s}px; height:{lh*s}px; z-index:1; }}
.who img {{ width:100%; height:100%; display:block; }} .who svg {{ position:absolute; inset:0; width:100%; height:100%; }}
h1 {{ position:absolute; left:86px; top:52px; font-size:{L['fs']}px; font-weight:500; letter-spacing:-4px; line-height:{L['lh']}; z-index:3; white-space:nowrap; }}
h1 b {{ font-weight:800; }} h1 sup {{ font-size:.45em; font-weight:500; vertical-align:.42em; margin-left:2px; }}
.old {{ position:relative; display:inline-block; margin-right:.12em; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}

.phone {{ position:absolute; left:{px}px; top:{py}px; width:{pw}px; height:{ph}px; transform:rotate({rot}deg); z-index:4;
          border-radius:72px; background:#111; padding:10px; box-shadow:0 50px 90px rgba(10,40,70,.28), 0 10px 24px rgba(10,40,70,.18); }}
.screen {{ position:relative; width:100%; height:100%; border-radius:62px; overflow:hidden; background:#F4EFE7; font-family:Inter; }}
.island {{ position:absolute; left:50%; top:16px; transform:translateX(-50%); width:150px; height:40px; border-radius:22px; background:#111; }}
.status {{ height:66px; background:#fff; display:flex; align-items:center; justify-content:space-between; padding:10px 44px 0; font-weight:700; font-size:22px; }}
.head {{ height:96px; background:#fff; display:flex; align-items:center; gap:16px; padding:0 26px; border-bottom:1px solid #ebe5dc; }}
.head .av {{ width:60px; height:60px; }}
.head .name {{ font-weight:700; font-size:26px; line-height:1.1; }} .head .name small {{ display:block; font-weight:500; font-size:18px; color:#1FA855; }}
.av {{ border-radius:50%; flex:none; background:url(data:image/png;base64,{av}) center/cover; }}
.chat {{ padding:26px 24px 0 20px; font-size:25px; }}
.row {{ display:flex; align-items:flex-end; gap:10px; margin-bottom:16px; }} .row.out {{ justify-content:flex-end; }}
.row .av {{ width:46px; height:46px; }}
.bub {{ background:#fff; border-radius:24px 24px 24px 8px; padding:14px 20px; line-height:1.3; color:#111; box-shadow:0 2px 4px rgba(0,0,0,.06); white-space:nowrap; }}
.voice {{ display:flex; align-items:center; gap:12px; height:72px; padding:0 22px; border-radius:28px 28px 8px 28px; background:#1FBE5A; color:#fff; }}
.voice .wave {{ height:28px; gap:3px; display:flex; align-items:center; }} .voice .wave i {{ display:block; width:3.4px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:20px; }}
.qrwrap {{ position:absolute; left:0; right:0; bottom:110px; display:flex; flex-direction:column; align-items:center; gap:14px; }}
.qrlabel {{ font-size:20px; font-weight:600; color:#8a847b; }}
.qr {{ width:{qs}px; height:{qs}px; background:#fff; border-radius:34px; box-shadow:0 10px 28px rgba(0,0,0,.12); display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:{qs*.88}px; height:{qs*.88}px; shape-rendering:crispEdges; }}
.input {{ position:absolute; left:0; right:0; bottom:0; height:96px; display:flex; align-items:flex-start; gap:16px; padding:16px 26px 0; }}
.input .plus {{ font-size:40px; line-height:52px; font-weight:300; }}
.input .field {{ flex:1; height:54px; border-radius:27px; background:#E9E4DC; color:#a39d94; font-size:22px; display:flex; align-items:center; padding:0 22px; }}
.home {{ position:absolute; bottom:12px; left:50%; transform:translateX(-50%); width:170px; height:6px; border-radius:3px; background:#111; }}

.scan {{ position:absolute; left:{nx}px; top:{ny}px; font-family:Gochi; font-size:46px; line-height:1.0; text-align:center; transform:rotate(-8deg); white-space:nowrap; z-index:5; }}
.scan .wa {{ width:44px; height:44px; vertical-align:-9px; margin:0 4px; }}
.scan svg.arr {{ position:absolute; right:-170px; top:-30px; }}
.bchat {{ position:absolute; right:50px; font-family:Inter; z-index:4; }}
.bchat .row {{ gap:.4em; margin-bottom:.6em; }} .bchat .av {{ width:2.1em; height:2.1em; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bchat .bub {{ padding:.55em .85em; border-radius:1em 1em 1em .3em; box-shadow:0 10px 26px rgba(20,40,90,.18); }}
.bchat .voice {{ height:2.9em; padding:0 1em; gap:.5em; border-radius:1.1em 1.1em .3em 1.1em; box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.bchat .voice .t {{ font-size:.85em; }}
.bqr {{ position:absolute; background:#fff; border-radius:40px; box-shadow:0 18px 44px rgba(20,40,90,.24); display:flex; align-items:center; justify-content:center; z-index:4; }}
.bqr svg {{ width:87%; height:87%; shape-rendering:crispEdges; }}
.bscan {{ position:absolute; font-family:Gochi; font-size:40px; line-height:1.0; text-align:center; transform:rotate(-8deg); white-space:nowrap; z-index:5; }}
.bscan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.barr {{ position:absolute; z-index:5; }}
.logo {{ position:absolute; left:34px; top:1330px; height:100px; z-index:5; }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; z-index:5; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; z-index:5; }}
</style></head><body><div class="p">
<div class="who"><img src="data:image/png;base64,{b64(layer)}"><svg viewBox="0 0 {lw} {lh}">{W['header']}</svg></div>
<h1>{HEADS[L['head']](W['old'])}</h1>
{bubbles if L['mode']=='bubbles' else phone_html}
{'' if L['mode']=='bubbles' else scan_html}
<img class="logo" src="{LOGO}" alt="Delos">
<div class="url">www.delos.so</div><div class="tag">*Vos employés IA autonomes sur mesure.</div>
</div></body></html>"""


if __name__ == '__main__':
    for f in H.glob('*.html'):
        f.unlink()
    for n in LAYOUTS:
        (H / f'{n}.html').write_text(page(n))
    print('ok')

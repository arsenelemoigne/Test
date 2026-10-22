"""Phone-free posters: worker cut-out + floating WhatsApp bubbles + big QR. Two headline variants per worker."""
import base64, pathlib, sys
from PIL import Image
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
from common import LOGO, QR_SVG, WA_ICON, wave, play

b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
GOCHI = b64(H.parent / 'assets/GochiHand.woff2')
RED = '#E8412F'

KATE_HEADER = """<g transform="rotate(2.4 98 85)"><circle cx="98" cy="88" r="44" fill="#cfcbca"/><circle cx="98" cy="74" r="15" fill="#f3f1f0"/>
<path d="M70 112 a28 22 0 0 1 56 0" fill="#f3f1f0"/><rect x="168" y="66" width="250" height="14" rx="7" fill="#5f6370"/>
<rect x="168" y="98" width="120" height="13" rx="6.5" fill="#b4b1b0"/></g>"""

JAMES_BG = """linear-gradient(180deg,#D9CEF3 0%,#CFC3F0 19%,#B4A6EA 21%,#9F92E2 36%,#7F76CF 70%,#6A63C1 100%)"""
JAMES_FX = """<div style="position:absolute;inset:0;background:
 linear-gradient(100deg,transparent 30%,rgba(255,255,255,.16) 33%,transparent 37%),
 linear-gradient(80deg,transparent 52%,rgba(255,255,255,.12) 55%,transparent 60%),
 linear-gradient(95deg,transparent 64%,rgba(255,255,255,.10) 67%,transparent 71%);filter:blur(6px)"></div>
<svg style="position:absolute;inset:0" viewBox="0 0 2000 1500">
<path d="M0 300 C 200 286 360 312 560 298 S 920 284 1120 300 S 1500 312 1700 296 S 1920 290 2000 300" stroke="#fff" stroke-opacity=".45" stroke-width="5" fill="none"/>
""" + "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width="2"/>'
              for x, y, r in [(1240, 520, 9), (1262, 470, 6), (1230, 430, 4), (420, 760, 7), (440, 720, 4), (1360, 980, 8), (980, 470, 5), (300, 1060, 6)]) + "</svg>"

WORKERS = {
    'kate': dict(name='Kate', old='boulot', bg='#C2E1EF', fx='', header=KATE_HEADER,
                 msgs=('Salut ! Moi c’est Kate 👋<br>Je gère tes réseaux !', 'C’est noté ! Ton post<br>part demain à 9h 🚀')),
    'sophie': dict(name='Sophie', old='prospection', bg='#7EB4C8', fx='', header='',
                   msgs=('Salut ! Moi c’est Sophie 👋<br>Je te trouve des clients !', 'Top ! 3 rendez-vous<br>calés cette semaine 🎣')),
    'james': dict(name='James', old='mails', bg=JAMES_BG, fx=JAMES_FX, header='',
                  msgs=('Salut ! Moi c’est James 👋<br>Je gère tes mails !', 'C’est fait ! Inbox à zéro<br>et relances envoyées 📬')),
}
# per worker / variant: layer scale + position, chat column left + top
LAYOUT = {
    ('kate', 'gauche'): dict(s=.86, left=690, top=330, col=1540, mt=560),
    ('kate', 'haut'): dict(s=.88, left=330, top=392, col=1440, mt=440),
    ('sophie', 'gauche'): dict(s=.8, left=0, top=None, col=1460, mt=462, foot_right=True),
    ('sophie', 'haut'): dict(s=.95, left=0, top=None, col=1460, mt=430, foot_right=True),
    ('james', 'gauche'): dict(s=.76, left=470, top=650, col=1500, mt=520),
    ('james', 'haut'): dict(s=.86, left=280, top=540, col=1460, mt=440),
}

STRIKE = f"""<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<path d="M0 58 C 20 46, 40 64, 60 50 S 90 44, 100 54" stroke="{RED}" stroke-width="9" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
<path d="M3 66 C 25 54, 45 72, 70 58 S 92 52, 98 62" stroke="{RED}" stroke-width="6" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>"""


def headline(variant, old):
    old_html = f'<span class="old">{old}{STRIKE}<span class="note-red">← méthode boomer</span></span>'
    if variant == 'gauche':
        return f'<h1 class="stack">Métro,<br>{old_html}<br><b>worker<sup>*</sup>,</b><br>dodo.</h1>'
    return f'<h1 class="top">Métro, {old_html}<br>worker<sup>*</sup>, dodo.</h1>'


def page(key, variant, headline_html=None, bg=None, msgs=None, layout=None):
    w = WORKERS[key]
    L = {**LAYOUT[(key, variant)], **(layout or {})}
    layer = H / f'layers/{key}.png'
    lw, lh = Image.open(layer).size
    s = L['s']
    top = L['top'] if L['top'] is not None else 1500 - lh * s
    av = b64(H / f'layers/{key}_avatar.png')
    m1, m2 = msgs or w['msgs']
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Métro worker dodo</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:{bg or w['bg']}; overflow:hidden; }}
.who {{ position:absolute; left:{L['left']}px; top:{top}px; width:{lw*s}px; height:{lh*s}px; }}
.who img {{ width:100%; height:100%; display:block; }}
.who svg {{ position:absolute; inset:0; width:100%; height:100%; }}
h1 {{ position:absolute; left:86px; top:56px; color:#000; z-index:3; }}
h1.stack {{ font-size:160px; font-weight:500; letter-spacing:-4.5px; line-height:.98; }}
h1.top {{ font-size:146px; font-weight:800; letter-spacing:-4.5px; line-height:1.0; }}
h1 b {{ font-weight:800; }}
h1.mix {{ top:84px; font-size:var(--fs,112px); font-weight:500; letter-spacing:-3px; line-height:1.06; width:1840px; }}
h1 sup {{ font-size:.45em; font-weight:500; vertical-align:.42em; margin-left:2px; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note-red {{ position:absolute; left:calc(100% + 28px); top:26%; font-family:Gochi; font-size:52px; font-weight:400; letter-spacing:0;
            color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
.chat {{ position:absolute; left:{L['col']}px; top:{L['mt']}px; width:{1950-L['col']}px; font-family:Inter; z-index:4; }}
.row {{ display:flex; align-items:flex-end; gap:10px; margin-bottom:14px; }}
.row.out {{ justify-content:flex-end; }}
.av {{ width:58px; height:58px; border-radius:50%; flex:none; background:url(data:image/png;base64,{av}) center/cover; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bub {{ background:#fff; border-radius:26px 26px 26px 8px; padding:15px 22px; font-size:25px; line-height:1.3; color:#111; white-space:nowrap;
        box-shadow:0 10px 26px rgba(20,40,90,.18); }}
.voice {{ display:flex; align-items:center; gap:14px; height:80px; padding:0 26px; border-radius:30px 30px 8px 30px; background:#1FBE5A; color:#fff;
          box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.voice .wave {{ height:30px; gap:3px; display:flex; align-items:center; }}
.voice .wave i {{ display:block; width:3.6px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:22px; }}
.qr {{ margin:24px 0 0 auto; width:400px; height:400px; background:#fff; border-radius:40px; box-shadow:0 18px 44px rgba(20,40,90,.24);
       display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:350px; height:350px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; font-family:Gochi; font-size:40px; line-height:1.0; text-align:center; transform:rotate(-8deg); white-space:nowrap; z-index:4; }}
.scan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:34px; top:1290px; height:108px; z-index:3; }}
.foot {{ position:absolute; left:34px; top:1418px; font-size:32px; color:#111; z-index:3; line-height:1.25; }}
.foot small {{ font-size:23px; }}
.foot.right {{ left:auto; right:50px; top:auto; bottom:178px; text-align:right; }}
.logo {{ filter:drop-shadow(0 0 18px rgba(255,255,255,.35)); }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; z-index:3; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; z-index:3; }}
</style></head><body><div class="p">
{w['fx']}
<div class="who"><img src="data:image/png;base64,{b64(layer)}"><svg viewBox="0 0 {lw} {lh}">{w['header']}</svg></div>
{headline_html or headline(variant, w['old'])}
<div class="chat">
  <div class="row"><div class="av"></div><div class="bub">{m1}</div></div>
  <div class="row out"><div class="voice">{play('#fff')}<span style="width:10px;height:10px;border-radius:50%;background:#fff"></span>{wave('rgba(255,255,255,.9)', 24)}<span class="t">0:32</span></div></div>
  <div class="row"><div class="av"></div><div class="bub">{m2}</div></div>
  <div class="qr">{QR_SVG}</div>
</div>
<div class="scan" id="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp</div>
<svg id="arrow" style="position:absolute;z-index:4" width="170" height="130" viewBox="0 0 170 130"><path d="M12 120 C 24 60 70 34 148 40" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/>
<path d="M114 10 L152 40 L112 70" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>
<img class="logo" src="{LOGO}" alt="Delos">
<div class="foot{' right' if L.get('foot_right') else ''}">*worker = employé<br><small>**boomer = un état d'esprit, pas une date de naissance</small></div>
<div class="url">www.delos.so</div>
<div class="tag">Vos employés IA autonomes sur mesure.</div>
</div><script>
const q=document.querySelector('.qr').getBoundingClientRect(), sc=document.getElementById('scan'), ar=document.getElementById('arrow');
sc.style.left=(q.left-sc.offsetWidth-30)+'px'; sc.style.top=(q.bottom-sc.offsetHeight-10)+'px';
ar.style.left=(q.left-175)+'px'; ar.style.top=(q.bottom-sc.offsetHeight-150)+'px';
</script></body></html>"""


if __name__ == '__main__':
    for f in H.glob('*.html'):
        f.unlink()
    for key in WORKERS:
        for v in ('gauche', 'haut'):
            (H / f'{key}-{v}.html').write_text(page(key, v))
    print('ok')

"""Creative variations: 'Mon neveu' series, typographic manifesto, vertical story."""
import base64, importlib.util, pathlib, sys
from PIL import Image
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
_spec = importlib.util.spec_from_file_location('v3build', H.parent / 'v3/build.py')
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)
from common import LOGO, QR_SVG, WA_ICON, wave, play

RED, b64 = v3.RED, v3.b64
GOCHI = v3.GOCHI


def strike(text, note='← méthode boomer', k=0):
    paths = [("M0 58 C 20 46, 40 64, 60 50 S 90 44, 100 54", "M3 66 C 25 54, 45 72, 70 58 S 92 52, 98 62"),
             ("M0 52 C 25 62, 45 44, 70 56 S 92 60, 100 48", "M2 60 C 30 70, 50 50, 72 62 S 94 64, 99 56"),
             ("M0 60 C 30 50, 55 66, 75 52 S 95 50, 100 58", "M4 54 C 28 64, 48 48, 68 60 S 90 62, 98 50")][k % 3]
    svg = (f'<svg viewBox="0 0 100 100" preserveAspectRatio="none"><path d="{paths[0]}" stroke="{RED}" stroke-width="9" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/>'
           f'<path d="{paths[1]}" stroke="{RED}" stroke-width="6" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>')
    n = f'<span class="note-red">{note}</span>' if note else ''
    return f'<span class="old">{text}{svg}{n}</span>'


# ---------- 1. "Mon neveu" series: the boomer workaround, crossed out
SERIES = {
    'neveu-kate': ('kate', strike('Mon neveu') + '<br><b>Mon worker<sup>*</sup></b> gère mes réseaux.', '#F6DC8E', 132,
                   ('Salut ! Moi c’est Kate 👋<br>Ton neveu peut souffler.', 'Post publié ✅<br>+1 200 vues ce matin 🚀')),
    'neveu-sophie': ('sophie', strike('Les Pages Jaunes') + '<br><b>Mon worker<sup>*</sup></b> me trouve des clients.', None, 114,
                     ('Salut ! Moi c’est Sophie 👋<br>Les Pages Jaunes, c’est fini.', '3 prospects chauds<br>et 2 rendez-vous 🎣')),
    'neveu-james': ('james', strike('Je vous mets en copie.') + '<br><b>Mon worker<sup>*</sup></b> répond à mes mails.', None, 124,
                    ('Salut ! Moi c’est James 👋<br>Plus personne en copie.', '47 mails traités,<br>inbox à zéro 📬')),
}

# ---------- 2. Typographic manifesto with a group chat of the three workers
def manifesto():
    olds = ['Métro.', 'Boulot.', 'Fax.', 'Excel_v12.', 'Réu de 2h.']
    lines = ''.join(f'<div>{strike(o, None, i)}</div>' for i, o in enumerate(olds))
    avs = {k: b64(v3.H / f'layers/{k}_avatar.png') for k in ('kate', 'sophie', 'james')}
    bub = lambda k, name, txt: (f'<div class="row"><div class="av" style="background-image:url(data:image/png;base64,{avs[k]})"></div>'
                                f'<div class="bub"><span class="who">Worker {name}</span>{txt}</div></div>')
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Manifeste worker</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:#D6CCF2; overflow:hidden; }}
.list {{ position:absolute; left:86px; top:50px; font-size:150px; font-weight:500; letter-spacing:-5px; line-height:1.0; }}
.list b {{ font-weight:900; font-size:196px; letter-spacing:-7px; display:block; margin-top:14px; }}
.list sup {{ font-size:.42em; font-weight:500; vertical-align:.45em; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.boomer {{ position:absolute; left:470px; top:380px; font-family:Gochi; font-size:56px; color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
.boomer svg {{ display:block; margin:6px 0 0 -20px; }}
.chat {{ position:absolute; left:1120px; top:70px; width:830px; font-family:Inter; }}
.title {{ font-family:Inter; font-weight:700; font-size:26px; color:#5b5480; margin:0 0 18px 74px; letter-spacing:.5px; }}
.row {{ display:flex; align-items:flex-end; gap:12px; margin-bottom:14px; }}
.row.out {{ justify-content:flex-end; }}
.av {{ width:70px; height:70px; border-radius:50%; flex:none; background:center/cover; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bub {{ background:#fff; border-radius:28px 28px 28px 8px; padding:16px 26px; font-size:31px; line-height:1.3; color:#111; box-shadow:0 10px 26px rgba(40,20,90,.16); }}
.bub .who {{ display:block; font-size:21px; font-weight:700; color:#1FA855; margin-bottom:2px; }}
.voice {{ display:flex; align-items:center; gap:14px; height:80px; padding:0 26px; border-radius:30px 30px 8px 30px; background:#1FBE5A; color:#fff; box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.voice .wave {{ height:30px; gap:3px; display:flex; align-items:center; }} .voice .wave i {{ display:block; width:3.6px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:22px; }}
.qr {{ position:absolute; left:1490px; top:740px; width:460px; height:460px; background:#fff; border-radius:42px; box-shadow:0 18px 44px rgba(40,20,90,.22);
       display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:410px; height:410px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; left:1060px; top:1010px; font-family:Gochi; font-size:42px; line-height:1; text-align:center; transform:rotate(-8deg); }}
.scan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:34px; top:1290px; height:108px; }}
.foot {{ position:absolute; left:34px; top:1418px; font-size:32px; color:#111; line-height:1.25; }} .foot small {{ font-size:23px; }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; }}
</style></head><body><div class="p">
<div class="list">{lines}<b>Worker<sup>*</sup>.</b></div>
<div class="boomer">← la méthode boomer</div>
<div class="chat"><div class="title">👥  MES WORKERS · en ligne</div>
{bub('james', 'James', 'Tes 47 mails sont traités 📬')}
{bub('sophie', 'Sophie', '2 rendez-vous calés jeudi 🎣')}
{bub('kate', 'Kate', 'Ton post cartonne sur LinkedIn 🚀')}
<div class="row out"><div class="voice">{play('#fff')}<span style="width:10px;height:10px;border-radius:50%;background:#fff"></span>{wave('rgba(255,255,255,.9)', 24)}<span class="t">0:09</span></div></div>
</div>
<div class="qr">{QR_SVG}</div>
<div class="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp
<svg width="140" height="110" viewBox="0 0 140 110" style="position:absolute;right:-150px;top:-40px"><path d="M10 100 C 30 40 70 24 126 30" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/><path d="M94 4 L130 30 L92 58" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
<img class="logo" src="{LOGO}" alt="Delos">
<div class="foot">*worker = employé<br><small>**boomer = un état d'esprit, pas une date de naissance</small></div>
<div class="url">www.delos.so</div><div class="tag">Vos employés IA autonomes sur mesure.</div>
</div></body></html>"""


# ---------- 3. Vertical story (1080x1920) with Kate
def story():
    lw, lh = Image.open(v3.H / 'layers/kate.png').size
    s, left, top = .68, -20, 650
    av = b64(v3.H / 'layers/kate_avatar.png')
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Story worker</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:1080px; height:1920px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:1080px; height:1920px; background:#C2E1EF; overflow:hidden; }}
.who {{ position:absolute; left:{left}px; top:{top}px; width:{lw*s}px; height:{lh*s}px; }}
.who img {{ width:100%; height:100%; display:block; }} .who svg {{ position:absolute; inset:0; width:100%; height:100%; }}
h1 {{ position:absolute; left:64px; top:84px; font-size:138px; font-weight:500; letter-spacing:-5px; line-height:.98; }}
h1 b {{ font-weight:800; }} h1 sup {{ font-size:.45em; font-weight:500; vertical-align:.42em; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note-red {{ position:absolute; left:calc(100% + 24px); top:30%; font-family:Gochi; font-size:48px; font-weight:400; letter-spacing:0; color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
.chat {{ position:absolute; left:56px; top:1440px; width:560px; font-family:Inter; }}
.row {{ display:flex; align-items:flex-end; gap:10px; margin-bottom:14px; }} .row.out {{ justify-content:flex-end; }}
.av {{ width:56px; height:56px; border-radius:50%; flex:none; background:url(data:image/png;base64,{av}) center/cover; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bub {{ background:#fff; border-radius:26px 26px 26px 8px; padding:14px 20px; font-size:25px; line-height:1.3; white-space:nowrap; box-shadow:0 10px 26px rgba(20,40,90,.18); }}
.voice {{ display:flex; align-items:center; gap:12px; height:76px; padding:0 22px; border-radius:30px 30px 8px 30px; background:#1FBE5A; color:#fff; box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.voice .wave {{ height:28px; gap:3px; display:flex; align-items:center; }} .voice .wave i {{ display:block; width:3.4px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:21px; }}
.qr {{ position:absolute; right:56px; top:1420px; width:350px; height:350px; background:#fff; border-radius:40px; box-shadow:0 18px 44px rgba(20,40,90,.24);
       display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:310px; height:310px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; right:66px; top:1250px; font-family:Gochi; font-size:38px; line-height:1; text-align:center; transform:rotate(-6deg); }}
.scan .wa {{ width:36px; height:36px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:56px; top:1790px; height:80px; }}
.url {{ position:absolute; right:56px; top:1786px; font-size:52px; font-weight:800; letter-spacing:-1px; }}
.tag {{ position:absolute; right:56px; top:1846px; font-size:30px; font-weight:500; }}
.foot {{ position:absolute; left:58px; top:1884px; font-size:20px; color:#222; }}
</style></head><body><div class="p">
<div class="who"><img src="data:image/png;base64,{b64(v3.H / 'layers/kate.png')}"><svg viewBox="0 0 {lw} {lh}">{v3.KATE_HEADER}</svg></div>
<h1>Métro,<br>{strike('boulot', '← méthode<br>&nbsp;&nbsp;&nbsp;&nbsp;boomer')}<br><b>worker<sup>*</sup>,</b><br>dodo.</h1>
<div class="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp ↓</div>
<div class="chat">
 <div class="row"><div class="av"></div><div class="bub">Salut ! Moi c’est Kate 👋<br>Je gère tes réseaux !</div></div>
 <div class="row out"><div class="voice">{play('#fff')}<span style="width:9px;height:9px;border-radius:50%;background:#fff"></span>{wave('rgba(255,255,255,.9)', 20)}<span class="t">0:32</span></div></div>
 <div class="row"><div class="av"></div><div class="bub">C’est noté ! Ton post<br>part demain à 9h 🚀</div></div>
</div>
<div class="qr">{QR_SVG}</div>
<img class="logo" src="{LOGO}" alt="Delos"><div class="url">www.delos.so</div><div class="tag">Vos employés IA autonomes sur mesure.</div>
<div class="foot">*worker = employé · **boomer = un état d'esprit</div>
</div></body></html>"""


for f in H.glob('*.html'):
    f.unlink()
for name, (key, head, bg, fs, msgs) in SERIES.items():
    (H / f'{name}.html').write_text(v3.page(key, 'haut', headline_html=f'<h1 class="mix" style="--fs:{fs}px">{head}</h1>', bg=bg, msgs=msgs))
(H / 'manifeste.html').write_text(manifesto())
(H / 'story-kate.html').write_text(story())
print('ok')

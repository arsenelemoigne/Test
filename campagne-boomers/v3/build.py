"""Kate poster without the phone: floating WhatsApp bubbles + big QR under the dab hand."""
import base64, pathlib, sys
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
from common import LOGO, QR_SVG, WA_ICON, wave, play

b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
GOCHI = b64(H.parent / 'assets/GochiHand.woff2')
KATE = b64(H / 'layers/kate.png')
AV = b64(H / 'layers/kate_avatar.png')
RED = '#E8412F'
LW, LH = 1194, 1115            # layer size; hand tip at (1188, 75) in layer coords

STRIKE = f"""<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<path d="M0 58 C 20 46, 40 64, 60 50 S 90 44, 100 54" stroke="{RED}" stroke-width="9" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
<path d="M3 66 C 25 54, 45 72, 70 58 S 92 52, 98 62" stroke="{RED}" stroke-width="6" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>"""

HEADLINES = {
    'gauche': f"""<h1 class="stack">Métro,<br><span class="old">boulot{STRIKE}<span class="note-red">← méthode boomer</span></span><br><b>worker<sup>*</sup>,</b><br>dodo.</h1>""",
    'haut': f"""<h1 class="top">Métro, <span class="old">boulot{STRIKE}<span class="note-red">← méthode boomer</span></span><br>worker<sup>*</sup>, dodo.</h1>""",
}
LAYOUT = {   # kate scale/left, chat column left/top
    'gauche': dict(s=.8, left=745, top=366, col=1556, msg_top=548),
    'haut': dict(s=.82, left=380, top=384, col=1420, msg_top=505),
}

def page(variant):
    L = LAYOUT[variant]
    s, left = L['s'], L['left']
    top = L['top']
    col, mt = L['col'], L['msg_top']
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Métro worker dodo</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:#C2E1EF; overflow:hidden; }}
.kate {{ position:absolute; left:{left}px; top:{top}px; width:{LW*s}px; height:{LH*s}px; }}
.kate img {{ width:100%; height:100%; display:block; }}
.kate svg {{ position:absolute; inset:0; width:100%; height:100%; }}
h1 {{ position:absolute; left:86px; top:56px; color:#000; line-height:.98; z-index:3; }}
h1.stack {{ font-size:160px; font-weight:500; letter-spacing:-4.5px; }}
h1.top {{ font-size:146px; font-weight:800; letter-spacing:-4.5px; line-height:1.0; }}
h1 b {{ font-weight:800; }}
h1 sup {{ font-size:.45em; font-weight:500; vertical-align:.42em; margin-left:2px; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note-red {{ position:absolute; left:calc(100% + 28px); top:26%; font-family:Gochi; font-size:52px; font-weight:400; letter-spacing:0;
            color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
.note-red.below {{ left:auto; right:-20px; top:auto; bottom:-62px; }}
.chat {{ position:absolute; left:{col}px; top:{mt}px; width:{1950-col}px; font-family:Inter; z-index:4; }}
.row {{ display:flex; align-items:flex-end; gap:10px; margin-bottom:14px; }}
.row.out {{ justify-content:flex-end; }}
.av {{ width:56px; height:56px; border-radius:50%; flex:none; background:url(data:image/png;base64,{AV}) center/cover; box-shadow:0 4px 10px rgba(0,0,0,.12); }}
.bub {{ background:#fff; border-radius:26px 26px 26px 8px; padding:15px 22px; font-size:24px; white-space:nowrap; line-height:1.3; color:#111;
        box-shadow:0 10px 26px rgba(20,60,90,.16); }}
.voice {{ display:flex; align-items:center; gap:14px; height:78px; padding:0 24px; border-radius:30px 30px 8px 30px; background:#1FBE5A; color:#fff;
          box-shadow:0 10px 26px rgba(20,90,50,.22); }}
.voice .wave {{ height:30px; gap:3px; display:flex; align-items:center; }}
.voice .wave i {{ display:block; width:3.4px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:21px; }}
.qr {{ margin:22px auto 0; width:320px; height:320px; background:#fff; border-radius:34px; box-shadow:0 18px 40px rgba(20,60,90,.2);
       display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:286px; height:286px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; font-family:Gochi; font-size:36px; line-height:1.0; text-align:center; transform:rotate(-8deg); white-space:nowrap; z-index:4; }}
.scan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:34px; top:1290px; height:108px; z-index:3; }}
.foot {{ position:absolute; left:34px; top:1418px; font-size:32px; color:#1a1a1a; z-index:3; line-height:1.25; }}
.foot small {{ font-size:23px; color:#333; }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; z-index:3; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; z-index:3; }}
</style></head><body><div class="p">
<div class="kate"><img src="data:image/png;base64,{KATE}">
<svg viewBox="0 0 {LW} {LH}"><g transform="rotate(2.4 98 85)">
<circle cx="98" cy="88" r="44" fill="#cfcbca"/><circle cx="98" cy="74" r="15" fill="#f3f1f0"/><path d="M70 112 a28 22 0 0 1 56 0" fill="#f3f1f0"/>
<rect x="168" y="66" width="250" height="14" rx="7" fill="#5f6370"/><rect x="168" y="98" width="120" height="13" rx="6.5" fill="#b4b1b0"/></g></svg></div>
{HEADLINES[variant]}
<div class="chat">
  <div class="row"><div class="av"></div><div class="bub">Salut ! Moi c’est Kate 👋<br>Je gère tes réseaux !</div></div>
  <div class="row out"><div class="voice">{play('#fff')}<span style="width:10px;height:10px;border-radius:50%;background:#fff"></span>{wave('rgba(255,255,255,.9)', 24)}<span class="t">0:32</span></div></div>
  <div class="row"><div class="av"></div><div class="bub">C’est noté ! Ton post<br>part demain à 9h 🚀</div></div>
  <div class="qr">{QR_SVG}</div>
</div>
<div class="scan" id="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp</div>
<svg id="arrow" style="position:absolute;z-index:4" width="150" height="150" viewBox="0 0 150 150"><path d="M40 140 C 10 100 30 50 118 34" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/>
<path d="M88 8 L124 34 L88 62" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>
<img class="logo" src="{LOGO}" alt="Delos">
<div class="foot">*worker = employé<br><small>**boomer = un état d'esprit, pas une date de naissance</small></div>
<div class="url">www.delos.so</div>
<div class="tag">Vos employés IA autonomes sur mesure.</div>
</div><script>
const q=document.querySelector('.qr').getBoundingClientRect(), sc=document.getElementById('scan'), ar=document.getElementById('arrow');
sc.style.left=(q.left+q.width/2-sc.offsetWidth/2-40)+'px'; sc.style.top=(q.bottom+6)+'px';
ar.style.left=(q.left-150)+'px'; ar.style.top=(q.bottom-70)+'px';
</script></body></html>"""

for v in HEADLINES:
    (H / f'kate-{v}.html').write_text(page(v))
print('ok')

"""Awareness series: Delos = the simple, reliable AI for companies."""
import importlib.util, pathlib, sys
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load('v3build', H.parent / 'v3/build.py')
v4 = _load('v4build', H.parent / 'v4/build.py')
from common import LOGO, QR_SVG, WA_ICON, wave, play

TAG = 'L’IA simple et fiable pour votre entreprise.'
FOOT = '*worker = employé'
strike, RED, GOCHI, b64 = v4.strike, v3.RED, v3.GOCHI, v3.b64

# ---------- character posters
CHAR = {
    'simple-vocal-james': dict(key='james', fs=118, voice='0:12',
        head='Si vous savez envoyer un vocal,<br>vous savez utiliser <b>Delos.</b>',
        msgs=('Salut ! Moi c’est James 👋<br>Dis-moi ce dont tu as besoin.', 'C’est fait ✅<br>Je t’envoie le récap.')),
    'fiable-fax-sophie': dict(key='sophie', fs=120, voice='0:21',
        head='Aussi fiable que votre vieux fax.<br><b>Sauf que lui, il répond.</b>',
        msgs=('Salut ! Moi c’est Sophie 👋<br>Je bosse même le dimanche.', 'Relance envoyée ✅<br>Rendez-vous jeudi 10h.')),
    'simple-14-outils-kate': dict(key='kate', fs=126, voice='0:08',
        head=strike('14 abonnements IA', '← trop compliqué') + '<br><b>1 worker<sup>*</sup></b>, sur WhatsApp.',
        msgs=('Salut ! Moi c’est Kate 👋<br>Pas d’appli, pas de formation.', 'Compris ✅<br>Je m’en occupe.')),
}


# ---------- typographic posters (left: statement, right: chat + QR)
def typo(title, left_html, rows_html, bg, extra_css=''):
    avs = {k: b64(v3.H / f'layers/{k}_avatar.png') for k in ('kate', 'sophie', 'james')}
    css = f"""
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{GOCHI}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:{bg}; overflow:hidden; }}
.left {{ position:absolute; left:86px; top:60px; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note-red {{ position:absolute; left:calc(100% + 26px); top:24%; font-family:Gochi; font-size:52px; font-weight:400; letter-spacing:0; color:{RED}; transform:rotate(-6deg); white-space:nowrap; }}
sup {{ font-size:.42em; font-weight:500; vertical-align:.45em; }}
.chat {{ position:absolute; left:1130px; top:80px; width:820px; font-family:Inter; }}
.title {{ font-weight:700; font-size:26px; color:rgba(0,0,0,.55); margin:0 0 18px 82px; letter-spacing:.5px; }}
.row {{ display:flex; align-items:flex-end; gap:12px; margin-bottom:14px; }}
.row.out {{ justify-content:flex-end; }}
.av {{ width:70px; height:70px; border-radius:50%; flex:none; background:center/cover; box-shadow:0 4px 10px rgba(0,0,0,.14); }}
.bub {{ background:#fff; border-radius:28px 28px 28px 8px; padding:14px 24px; font-size:29px; line-height:1.3; color:#111; box-shadow:0 10px 26px rgba(20,20,60,.15); }}
.bub .time {{ display:block; font-size:19px; font-weight:700; color:#1FA855; margin-bottom:2px; }}
.voice {{ display:flex; align-items:center; gap:14px; height:82px; padding:0 26px; border-radius:30px 30px 8px 30px; background:#1FBE5A; color:#fff; box-shadow:0 10px 26px rgba(20,90,50,.24); }}
.voice .wave {{ height:30px; gap:3px; display:flex; align-items:center; }} .voice .wave i {{ display:block; width:3.6px; border-radius:2px; }}
.voice .t {{ font-weight:700; font-size:22px; }}
.qr {{ position:absolute; left:1500px; top:790px; width:450px; height:450px; background:#fff; border-radius:42px; box-shadow:0 18px 44px rgba(20,20,60,.2);
       display:flex; align-items:center; justify-content:center; }}
.qr svg {{ width:400px; height:400px; shape-rendering:crispEdges; }}
.scan {{ position:absolute; left:1080px; top:1040px; font-family:Gochi; font-size:42px; line-height:1; text-align:center; transform:rotate(-8deg); }}
.scan .wa {{ width:40px; height:40px; vertical-align:-8px; margin:0 4px; }}
.logo {{ position:absolute; left:34px; top:1290px; height:108px; }}
.foot {{ position:absolute; left:34px; top:1418px; font-size:32px; color:#111; }}
.url {{ position:absolute; right:50px; top:1330px; font-size:72px; font-weight:800; letter-spacing:-1px; }}
.tag {{ position:absolute; right:44px; top:1408px; font-size:58px; font-weight:500; letter-spacing:-1px; }}
{extra_css}"""
    rows = rows_html.format(**{f'av_{k}': f'style="background-image:url(data:image/png;base64,{v})"' for k, v in avs.items()},
                            voice=lambda t: '')
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body><div class="p">
<div class="left">{left_html}</div>
<div class="chat">{rows}</div>
<div class="qr">{QR_SVG}</div>
<div class="scan">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp
<svg width="140" height="110" viewBox="0 0 140 110" style="position:absolute;right:-150px;top:-40px"><path d="M10 100 C 30 40 70 24 126 30" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/><path d="M94 4 L130 30 L92 58" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
<img class="logo" src="{LOGO}" alt="Delos"><div class="foot">{FOOT}</div>
<div class="url">www.delos.so</div><div class="tag">{TAG}</div>
</div></body></html>"""


VOICE = lambda t: (f'<div class="row out"><div class="voice">{play("#fff")}<span style="width:10px;height:10px;border-radius:50%;background:#fff"></span>'
                   f'{wave("rgba(255,255,255,.9)", 24)}<span class="t">{t}</span></div></div>')


def log_poster():
    entries = [('Lun. 08:02', '32 mails triés, 4 urgents 📬'), ('Mer. 14:15', '3 rendez-vous calés 🗓️'),
               ('Ven. 18:47', 'Récap de la semaine envoyé 📄'), ('Dim. 23:10', 'Devis relancé, client OK ✅')]
    rows = '<div class="title">👤  WORKER JAMES · en ligne</div>' + ''.join(
        f'<div class="row"><div class="av" {{av_james}}></div><div class="bub"><span class="time">{t}</span>{m}</div></div>' for t, m in entries)
    left = ('<div style="font-size:150px;font-weight:500;letter-spacing:-5px;line-height:1.0">Lundi 8h.<br>Vendredi 19h.<br>Dimanche 23h.</div>'
            '<div style="font-size:196px;font-weight:900;letter-spacing:-7px;line-height:1.0;margin-top:30px">Toujours<br>là.</div>')
    return typo('Toujours là', left, rows, '#F6DC8E', '.chat{top:70px} .qr{top:600px} .scan{top:830px}')


def blabla_poster():
    jargon = ['LLM.', 'Prompt engineering.', 'POC de 6 mois.', 'Agents multimodaux.']
    left = ('<div style="font-size:118px;font-weight:500;letter-spacing:-4px;line-height:1.08">'
            + ''.join(f'<div>{strike(j, None, i)}</div>' for i, j in enumerate(jargon)) + '</div>'
            '<div style="font-size:176px;font-weight:900;letter-spacing:-6px;line-height:1.0;margin-top:40px">Un vocal.<br>C’est fait.</div>')
    rows = ('<div class="title">👤  WORKER KATE · en ligne</div>' + VOICE('0:14')
            + '<div class="row"><div class="av" {av_kate}></div><div class="bub">C’est fait ✅<br>Post publié, devis envoyé,<br>rendez-vous calé.</div></div>')
    return typo('Sans le blabla', left, rows, '#C2E1EF', '.chat{top:120px} .qr{top:640px} .scan{top:870px}')


if __name__ == '__main__':
    for f in H.glob('*.html'):
        f.unlink()
    for name, c in CHAR.items():
        (H / f'{name}.html').write_text(v3.page(c['key'], 'haut', headline_html=f'<h1 class="mix" style="--fs:{c["fs"]}px">{c["head"]}</h1>',
                                                msgs=c['msgs'], tag=TAG, foot=FOOT, voice=c['voice']))
    (H / 'fiable-toujours-la.html').write_text(log_poster())
    (H / 'simple-sans-blabla.html').write_text(blabla_poster())
    print('ok')

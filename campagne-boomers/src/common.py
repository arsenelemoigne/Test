"""Shared layout for the Delos "boomers" posters (2000x1500, same grid as the original campaign)."""
import base64
import html
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def _b64(path):
    return base64.b64encode(path.read_bytes()).decode()


LOGO = f"data:image/png;base64,{_b64(ASSETS / 'logo.png')}"
QR_SVG = (ASSETS / "qr.svg").read_text().split("?>", 1)[-1].replace('width="29" height="29"', 'viewBox="0 0 29 29"')
GOCHI = f"data:font/woff2;base64,{_b64(ASSETS / 'GochiHand.woff2')}"

CSS = """
@font-face { font-family: 'Gochi'; src: url(%(gochi)s) format('woff2'); }
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 2000px; height: 1500px; overflow: hidden; }
body { font-family: 'Inter Display', 'Inter', sans-serif; color: #000; -webkit-font-smoothing: antialiased; }
.poster { position: relative; width: 2000px; height: 1500px; overflow: hidden; }
.scene { position: absolute; inset: 0; width: 2000px; height: 1500px; }
h1 { position: absolute; left: 86px; top: 62px; width: 1660px; font-size: 124px; line-height: 1.07;
     letter-spacing: -2.5px; font-weight: 500; z-index: 5; }
h1 b { font-weight: 800; letter-spacing: -2px; }
h1 sup { font-size: 62px; font-weight: 500; vertical-align: 52px; margin-left: 6px; }
.logo { position: absolute; left: 34px; top: 1290px; height: 108px; z-index: 5; }
.foot { position: absolute; left: 34px; top: 1418px; font-size: 32px; font-weight: 400; color: #1a1a1a; z-index: 5; line-height: 1.25; }
.foot small { font-size: 23px; color: #333; }
.url { position: absolute; right: 50px; top: 1322px; font-size: 76px; font-weight: 800; letter-spacing: -1px; z-index: 5; }
.tag { position: absolute; right: 42px; top: 1404px; font-size: 64px; font-weight: 500; letter-spacing: -1px; z-index: 5; }

.phone { position: absolute; left: 1436px; top: 372px; width: 440px; height: 940px; border-radius: 70px;
         background: #111; padding: 9px; z-index: 6;
         box-shadow: 0 40px 70px rgba(0,0,0,.22), 0 8px 18px rgba(0,0,0,.15); }
.screen { position: relative; width: 100%%; height: 100%%; border-radius: 62px; overflow: hidden; background: #F6F2EB;
          font-family: 'Inter', sans-serif; }
.status { height: 58px; background: #fff; display: flex; align-items: center; justify-content: space-between; padding: 8px 34px 0 38px;
          font-weight: 700; font-size: 19px; }
.island { position: absolute; left: 50%%; top: 12px; transform: translateX(-50%%); width: 142px; height: 36px; background: #111; border-radius: 20px; }
.sig { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.bars { display: flex; align-items: flex-end; gap: 2px; height: 13px; }
.bars i { display: block; width: 3px; background: #111; border-radius: 1px; }
.batt { width: 27px; height: 13px; border: 1.6px solid #111; border-radius: 4px; padding: 1.5px; }
.batt::after { content: ''; display: block; width: 80%%; height: 100%%; background: #111; border-radius: 1.5px; }
.head { height: 66px; background: #fff; display: flex; align-items: center; padding: 0 22px; gap: 12px;
        border-bottom: 1px solid #ece7df; }
.head .name { font-weight: 700; font-size: 20px; flex: 1; }
.head .dots { font-size: 26px; font-weight: 800; letter-spacing: 1px; line-height: 0; }
.av { width: 44px; height: 44px; border-radius: 50%%; overflow: hidden; flex: none; }
.av svg { width: 100%%; height: 100%%; display: block; }
.chat { padding: 12px 16px 0 14px; }
.date { text-align: center; font-size: 14px; color: #8a847b; margin: 2px 0 12px; }
.row { display: flex; align-items: flex-end; gap: 8px; margin-bottom: 8px; }
.row.out { justify-content: flex-end; }
.row .av { width: 32px; height: 32px; }
.row .sp { width: 32px; flex: none; }
.bub { background: #fff; border-radius: 18px; padding: 11px 15px; font-size: 17px; line-height: 1.36; max-width: 330px;
       box-shadow: 0 1px 2px rgba(0,0,0,.06); color: #111; }
.bub.out { background: #DCF8C6; }
.voice { display: flex; align-items: center; gap: 10px; height: 60px; border-radius: 22px; padding: 0 18px; }
.voice.out { background: #1FBE5A; color: #fff; width: 236px; }
.voice.in { background: #fff; width: 236px; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
.voice .t { font-size: 15px; font-weight: 600; margin-left: auto; }
.wave { display: flex; align-items: center; gap: 2px; height: 22px; }
.wave i { display: block; width: 2.4px; border-radius: 2px; }
.vu { text-align: right; font-size: 12px; color: #8a847b; margin: -4px 4px 8px 0; }
.trans { font-size: 13px; color: #6b665e; font-style: italic; text-align: right; margin: -2px 6px 8px 0; }
.react { margin: -14px 0 6px 52px; width: 26px; height: 26px; border-radius: 50%%; background: #fff; display: flex; align-items: center;
         justify-content: center; font-size: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.12); position: relative; }
.qr { position: absolute; left: 92px; bottom: 102px; width: 296px; height: 296px; background: #fff; border-radius: 28px;
      box-shadow: 0 10px 30px rgba(0,0,0,.14); display: flex; align-items: center; justify-content: center; }
.qr svg { width: 250px; height: 250px; shape-rendering: crispEdges; }
.input { position: absolute; left: 0; right: 0; bottom: 0; height: 92px; display: flex; align-items: flex-start; padding: 12px 20px 0; gap: 14px; }
.input .plus { font-size: 34px; font-weight: 300; line-height: 44px; color: #222; }
.input .field { flex: 1; height: 46px; border-radius: 23px; background: #ECE8E1; display: flex; align-items: center; padding: 0 16px;
                color: #a29d95; font-size: 17px; justify-content: space-between; }
.home { position: absolute; bottom: 10px; left: 50%%; transform: translateX(-50%%); width: 150px; height: 5px; border-radius: 3px; background: #111; }

.note { position: absolute; left: 1040px; top: 1108px; width: 440px; white-space: nowrap; font-family: 'Gochi'; font-size: 42px; line-height: 1.02;
        text-align: center; transform: rotate(-8deg); z-index: 6; color: #111; }
.note .wa { display: inline-block; width: 38px; height: 38px; vertical-align: -8px; margin: 0 4px; }
.arrow { position: absolute; left: 1270px; top: 960px; width: 230px; height: 190px; z-index: 6; }
"""


def avatar(skin, hair, bg, style="short", extra=""):
    """Tiny flat-3D portrait used for the worker in the WhatsApp header."""
    hair_shapes = {
        "short": f'<path d="M14 40 C12 18 30 10 50 12 C70 10 88 20 86 42 C80 30 70 26 50 27 C34 26 22 30 14 40Z" fill="{hair}"/>',
        "long": f'<path d="M12 96 C4 60 10 16 50 12 C90 16 96 60 88 96 L76 96 C80 66 78 40 70 34 C60 30 40 30 30 34 C22 40 20 66 24 96Z" fill="{hair}"/>'
                f'<path d="M24 40 C30 22 70 18 78 40 C66 30 40 28 24 40Z" fill="{hair}"/>',
        "bun": f'<circle cx="50" cy="10" r="13" fill="{hair}"/><path d="M14 44 C12 18 32 12 50 13 C68 12 88 18 86 44 C78 30 66 27 50 28 C34 27 22 30 14 44Z" fill="{hair}"/>',
        "curly": "".join(f'<circle cx="{x}" cy="{y}" r="12" fill="{hair}"/>' for x, y in
                         [(20, 34), (28, 20), (42, 13), (58, 13), (72, 20), (80, 34), (16, 48), (84, 48)]),
    }[style]
    return f"""<svg viewBox="0 0 100 100"><rect width="100" height="100" fill="{bg}"/>
<ellipse cx="50" cy="104" rx="38" ry="24" fill="#fff" opacity=".9"/>
<rect x="42" y="62" width="16" height="18" rx="6" fill="{skin}"/>
<ellipse cx="50" cy="48" rx="24" ry="27" fill="{skin}"/>
<ellipse cx="41" cy="50" rx="3" ry="3.6" fill="#2a1b12"/><ellipse cx="59" cy="50" rx="3" ry="3.6" fill="#2a1b12"/>
<path d="M42 62 Q50 69 58 62" stroke="#7a3b2a" stroke-width="2.6" fill="none" stroke-linecap="round"/>
<ellipse cx="36" cy="59" rx="4.5" ry="2.6" fill="#f08d7a" opacity=".35"/><ellipse cx="64" cy="59" rx="4.5" ry="2.6" fill="#f08d7a" opacity=".35"/>
{extra}{hair_shapes}</svg>"""


def wave(color, n=30, seed=3):
    hs = [(((i * 7 + seed) * 37) % 17) + 4 for i in range(n)]
    return '<div class="wave">' + "".join(f'<i style="height:{h}px;background:{color}"></i>' for h in hs) + "</div>"


def play(color):
    return f'<svg width="18" height="20" viewBox="0 0 18 20"><path d="M2 1.5 L16 10 L2 18.5Z" fill="{color}" stroke="{color}" stroke-linejoin="round" stroke-width="2"/></svg>'


WA_ICON = """<svg class="wa" viewBox="0 0 40 40"><rect width="40" height="40" rx="9" fill="#25D366"/>
<path d="M20 8 A12 12 0 0 0 9.6 26 L8 32 L14.2 30.4 A12 12 0 1 0 20 8Z" fill="none" stroke="#fff" stroke-width="2.4"/>
<path d="M15.4 14.5 c.6-.6 1.4-.4 1.7.3 l1 2.2 c.2.5 0 1-.4 1.4 l-.6.6 c.8 1.6 2 2.8 3.6 3.6 l.6-.6 c.4-.4.9-.6 1.4-.4 l2.2 1 c.7.3.9 1.1.3 1.7 l-.9 .9 c-1 1-2.6 1.1-4 .4 c-2.8-1.4-5-3.6-6.4-6.4 c-.7-1.4-.6-3 .4-4Z" fill="#fff"/></svg>"""

ARROW = """<svg class="arrow" viewBox="0 0 230 190"><path d="M28 176 C 20 110 70 50 196 54" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round"/>
<path d="M160 22 L204 54 L156 84" fill="none" stroke="#151515" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def phone(name, av_svg, messages):
    rows = []
    for i, m in enumerate(messages):
        kind = m[0]
        nxt = messages[i + 1][0] if i + 1 < len(messages) else None
        is_in = kind.startswith("in")
        last_of_group = is_in and not (nxt or "").startswith("in")
        lead = (f'<div class="av">{av_svg}</div>' if last_of_group else '<div class="sp"></div>') if is_in else ""
        if kind == "in":
            rows.append(f'<div class="row">{lead}<div class="bub">{m[1]}</div></div>')
        elif kind == "out":
            rows.append(f'<div class="row out"><div class="bub out">{m[1]}</div></div>')
        elif kind == "out_voice":
            rows.append(f'<div class="row out"><div class="voice out">{play("#fff")}'
                        f'<span style="width:8px;height:8px;border-radius:50%;background:#fff;flex:none"></span>{wave("rgba(255,255,255,.85)", 26)}'
                        f'<span class="t">{m[1]}</span></div></div>')
            if len(m) > 2:
                rows.append(f'<div class="trans">{m[2]}</div>')
            else:
                rows.append('<div class="vu">Vu</div>')
        elif kind == "in_voice":
            rows.append(f'<div class="row">{lead}<div class="voice in">{play("#1FBE5A")}'
                        f'<span style="width:8px;height:8px;border-radius:50%;background:#1FBE5A;flex:none"></span>{wave("#b9b4ac", 26, 5)}'
                        f'<span class="t" style="color:#333">{m[1]}</span></div></div>')
            if len(m) > 2:
                rows.append(f'<div class="react">{m[2]}</div>')
    sig = '<div class="bars">' + "".join(f'<i style="height:{h}px"></i>' for h in (4, 7, 10, 13)) + "</div>"
    return f"""<div class="phone"><div class="screen">
<div class="island"></div>
<div class="status"><span>9:41 AM</span><span class="sig">{sig}<b>5G</b><span class="batt"></span></span></div>
<div class="head"><svg width="14" height="24" viewBox="0 0 14 24"><path d="M12 2 L3 12 L12 22" stroke="#111" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
<div class="av">{av_svg}</div><span class="name">{name}</span><span class="dots">···</span></div>
<div class="chat"><div class="date">Aujourd'hui 9:41 AM</div>{''.join(rows)}</div>
<div class="qr">{QR_SVG}</div>
<div class="input"><span class="plus">+</span><div class="field"><span>Message...</span>
<svg width="16" height="24" viewBox="0 0 16 24"><rect x="4" y="1" width="8" height="14" rx="4" fill="none" stroke="#555" stroke-width="2"/><path d="M1 11 a7 7 0 0 0 14 0 M8 18 v5" stroke="#555" stroke-width="2" fill="none" stroke-linecap="round"/></svg></div></div>
<div class="home"></div>
</div></div>"""


def poster(*, bg, question, scene_svg, worker, avatar_svg, messages, extra_html="", scene_transform=""):
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Delos — {html.escape(worker)}</title>
<style>{CSS % {'gochi': GOCHI}}</style></head><body>
<div class="poster" style="background:{bg}">
<svg class="scene" viewBox="0 0 2000 1500"><g transform="{scene_transform}">{scene_svg}</g></svg>
<h1>{question} <b>Parlez-en à votre&nbsp;worker<sup>*</sup>&#8239;!</b></h1>
{extra_html}
{phone(worker, avatar_svg, messages)}
{ARROW}
<div class="note">Scannez pour discuter<br>avec votre worker<br>via {WA_ICON}WhatsApp</div>
<img class="logo" src="{LOGO}" alt="Delos">
<div class="foot">*worker = employé<br><small>**boomer = un état d'esprit, pas une date de naissance</small></div>
<div class="url">www.delos.so</div>
<div class="tag">Vos employés IA autonomes sur mesure.</div>
</div></body></html>"""

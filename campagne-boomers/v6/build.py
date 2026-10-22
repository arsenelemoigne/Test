"""Funny series aimed at bored metro commuters (B2C reach) that lands with B2B decision makers."""
import importlib.util, pathlib, sys
H = pathlib.Path(__file__).parent
sys.path.insert(0, str(H.parent / 'src'))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v3 = _load('v3build', H.parent / 'v3/build.py')
v4 = _load('v4build', H.parent / 'v4/build.py')
v5 = _load('v5build', H.parent / 'v5/build.py')
strike, RED, TAG, FOOT = v4.strike, v3.RED, v5.TAG, v5.FOOT

PAIN_AU_CHOCOLAT = """<svg viewBox="0 0 900 520" width="900" height="520" style="overflow:visible">
<defs>
 <linearGradient id="crust" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F6C46A"/><stop offset=".5" stop-color="#DE9233"/><stop offset="1" stop-color="#A9581A"/></linearGradient>
 <linearGradient id="fold" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8E4A14" stop-opacity=".0"/><stop offset=".6" stop-color="#8E4A14" stop-opacity=".55"/><stop offset="1" stop-color="#8E4A14" stop-opacity="0"/></linearGradient>
 <radialGradient id="shine" cx=".4" cy=".15" r=".65"><stop offset="0" stop-color="#fff" stop-opacity=".6"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
 <linearGradient id="choc" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6b3418"/><stop offset="1" stop-color="#3a1a0a"/></linearGradient>
 <filter id="ds" x="-20%" y="-30%" width="140%" height="180%"><feDropShadow dx="0" dy="36" stdDeviation="30" flood-color="#6b3a08" flood-opacity=".35"/></filter>
</defs>
<g filter="url(#ds)">
 <rect x="40" y="170" width="120" height="58" rx="14" fill="url(#choc)"/><rect x="48" y="290" width="120" height="58" rx="14" fill="url(#choc)"/>
 <rect x="740" y="160" width="120" height="58" rx="14" fill="url(#choc)"/><rect x="732" y="280" width="120" height="58" rx="14" fill="url(#choc)"/>
 <path d="M120 120 Q 450 70 790 112 Q 830 250 790 400 Q 450 450 120 408 Q 86 260 120 120 Z" fill="url(#crust)"/>
 <path d="M112 200 Q 450 160 800 192 L 802 236 Q 450 206 110 246 Z" fill="url(#fold)"/>
 <path d="M106 300 Q 450 264 804 296 L 802 340 Q 450 312 108 348 Z" fill="url(#fold)"/>
 <path d="M150 140 Q 450 98 760 132" stroke="#FBDC9A" stroke-width="16" fill="none" stroke-linecap="round" opacity=".85"/>
 <path d="M150 262 Q 450 226 770 254" stroke="#F6C877" stroke-width="10" fill="none" stroke-linecap="round" opacity=".7"/>
 <path d="M150 374 Q 450 340 770 366" stroke="#E8A953" stroke-width="8" fill="none" stroke-linecap="round" opacity=".6"/>
 <path d="M120 120 Q 450 70 790 112 Q 830 250 790 400 Q 450 450 120 408 Q 86 260 120 120 Z" fill="url(#shine)"/>
 <g fill="#fff" opacity=".75"><circle cx="300" cy="130" r="6"/><circle cx="520" cy="116" r="5"/><circle cx="660" cy="150" r="6"/><circle cx="400" cy="250" r="4"/><circle cx="610" cy="280" r="4"/></g>
</g></svg>"""


def pain_quotidien():
    left = f"""<div style="font-size:150px;font-weight:500;letter-spacing:-5px;line-height:1.0">Delos.<br><b style="font-weight:900">Votre pain<br>quotidien.</b></div>
<div style="position:absolute;left:640px;top:440px;font-family:Gochi;font-size:60px;color:{RED};transform:rotate(-8deg);white-space:nowrap">(au chocolat)</div>
<div style="position:absolute;left:0px;top:640px;transform:rotate(-7deg)">{PAIN_AU_CHOCOLAT}</div>"""
    rows = ('<div class="title">👤  WORKER JAMES · en ligne</div>'
            '<div class="row"><div class="av" {av_james}></div><div class="bub"><span class="time">07:58</span>Bonjour ! Café, pain au chocolat<br>et récap du jour ? 🥐</div></div>'
            + v5.VOICE('0:05') +
            '<div class="row"><div class="av" {av_james}></div><div class="bub"><span class="time">07:59</span>Le récap, c’est fait ✅<br>Le pain au chocolat, je te laisse.</div></div>')
    return v5.typo('Votre pain quotidien', left, rows, '#F8DDA4', '.chat{top:90px} .qr{top:700px} .scan{top:930px}')


CHAR = {
    'chocolatine-sophie': dict(key='sophie', fs=112, voice='0:17',
        head='Pain au chocolat ou chocolatine&#8239;?<br><b>Votre worker<sup>*</sup></b> ne tranche pas.<br>Il bosse.',
        msgs=('Le débat, je te le laisse 🥐<br>Je m’occupe de tes clients.', '12 relances envoyées ✅<br>Toujours pas d’avis sur le pain.')),
    'metro-6-stations-kate': dict(key='kate', fs=124, voice='0:19',
        head='Encore 6 stations&#8239;?<br><b>Le temps de déléguer 3 tâches.</b>',
        msgs=('On a 6 stations 🚇<br>Dis-moi 3 trucs que tu détestes faire.', 'Noté ✅ Quand tu sors du métro,<br>c’est fait.')),
    'dsi-james': dict(key='james', fs=124, voice='0:11',
        head='Montrez ce QR code à votre DSI.<br><b>Il va adorer.</b> <span style="font-weight:400">(Ou paniquer.)</span>',
        msgs=('Salut le DSI 👋<br>On parle sécurité&#8239;?', 'Je t’envoie de quoi te rassurer 📎<br>Et une démo de 15 min.')),
}

if __name__ == '__main__':
    for f in H.glob('*.html'):
        f.unlink()
    (H / 'pain-quotidien.html').write_text(pain_quotidien())
    for name, c in CHAR.items():
        (H / f'{name}.html').write_text(v3.page(c['key'], 'haut', headline_html=f'<h1 class="mix" style="--fs:{c["fs"]}px">{c["head"]}</h1>',
                                                msgs=c['msgs'], tag=TAG, foot=FOOT, voice=c['voice']))
    print('ok')

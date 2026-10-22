"""Hand-built SVG scenes, one per poster. Coordinates are in poster space (2000x1500)."""

DEFS = """
<defs>
  <filter id="sh" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="28" stdDeviation="26" flood-color="#2b1a05" flood-opacity=".28"/></filter>
  <filter id="shs" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#2b1a05" flood-opacity=".22"/></filter>
  <filter id="blur"><feGaussianBlur stdDeviation="18"/></filter>
  <filter id="blur6"><feGaussianBlur stdDeviation="6"/></filter>
  <radialGradient id="light" cx=".35" cy=".25" r=".9"><stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
</defs>"""


def ground(cx, cy, rx, ry=46, op=.18):
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#000" opacity="{op}" filter="url(#blur)"/>'


def specks(points, color="#fff", op=.6):
    return "".join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" opacity="{op}"/>' for x, y, r in points)


def stamp(x, y, rot, text="100% BOOMER**", color="#D7263D", size=40):
    w = len(text) * size * .66 + 40
    return f"""<g transform="translate({x} {y}) rotate({rot})" opacity=".9">
<rect x="{-w/2}" y="-{size*.95}" width="{w}" height="{size*1.9}" rx="10" fill="none" stroke="{color}" stroke-width="6"/>
<rect x="{-w/2+9}" y="-{size*.95-9}" width="{w-18}" height="{size*1.9-18}" rx="6" fill="none" stroke="{color}" stroke-width="2.5"/>
<text x="0" y="{size*.36}" text-anchor="middle" font-family="Inter Display" font-weight="900" font-size="{size}" fill="{color}" letter-spacing="2">{text}</text></g>"""


# ---------------------------------------------------------------- FAX
def fax():
    keys = "".join(
        f'<rect x="{872 + c*52 - r*6}" y="{720 + r*24}" width="40" height="17" rx="7" fill="#d9cfb8" stroke="#bfb397" stroke-width="2"/>'
        for r in range(4) for c in range(3))
    grill = "".join(f'<circle cx="{372 + c*22}" cy="{965 + r*22}" r="5" fill="#a99a78"/>' for r in range(5) for c in range(9))
    web_anchor = (1072, 612)
    spokes = [(968, 486), (1010, 470), (1150, 520), (1186, 640), (1120, 720), (1000, 712), (948, 600)]
    web = "".join(f'<line x1="{web_anchor[0]}" y1="{web_anchor[1]}" x2="{x}" y2="{y}"/>' for x, y in spokes)
    rings = ""
    for k in (.25, .45, .65, .85):
        pts = " ".join(f"{web_anchor[0] + (x - web_anchor[0]) * k:.0f},{web_anchor[1] + (y - web_anchor[1]) * k:.0f}" for x, y in spokes)
        rings += f'<polyline points="{pts}" />'
    legs = "".join(
        f'<path d="M1182 642 q{s*22} {-14 + i*9} {s*34} {-2 + i*12}" stroke="#1a1a1a" stroke-width="4" fill="none" stroke-linecap="round"/>'
        for s in (-1, 1) for i in range(4))
    return DEFS + ground(720, 1236, 560) + f"""
<!-- calendar -->
<g transform="translate(130 520) rotate(-7)" filter="url(#shs)">
  <rect width="220" height="250" rx="16" fill="#fffdf8"/>
  <path d="M0 16 a16 16 0 0 1 16 -16 h188 a16 16 0 0 1 16 16 v54 h-220z" fill="#D7263D"/>
  <text x="110" y="50" text-anchor="middle" font-family="Inter Display" font-weight="800" font-size="30" fill="#fff" letter-spacing="2">JANVIER</text>
  <text x="110" y="160" text-anchor="middle" font-family="Inter Display" font-weight="900" font-size="84" fill="#1b1b1b">2009</text>
  <text x="110" y="214" text-anchor="middle" font-family="Gochi" font-size="25" fill="#2c4fb0">relancer le fax !!</text>
  <circle cx="60" cy="6" r="9" fill="#5a5a5a"/><circle cx="160" cy="6" r="9" fill="#5a5a5a"/>
</g>
<!-- paper sheet coming out of the fax -->
<g filter="url(#shs)">
  <path d="M560 700 C 556 610 580 530 640 478 L 968 478 C 905 532 882 612 884 700 Z" fill="#fbfaf6"/>
  <path d="M640 478 C 650 440 700 430 720 452 L 1046 452 C 1020 432 980 440 968 478 Z" fill="#e9e5dc"/>
  <g font-family="DejaVu Sans Mono" fill="#4a4a4a" transform="translate(612 548) skewX(-12)">
    <text font-size="25" font-weight="700" x="18" y="0">TRANSMISSION</text>
    <text font-size="25" font-weight="700" x="40" y="34">ÉCHOUÉE</text>
    <text font-size="18" x="10" y="78">Page 1/47 ...........</text>
    <text font-size="18" x="0" y="104">Réessayer : 3 812 fois</text>
  </g>
</g>
<!-- fax body -->
<g filter="url(#sh)">
  <path d="M370 690 h700 q22 0 32 18 l66 118 h-896 l66 -118 q10 -18 32 -18z" fill="#f4eedf"/>
  <path d="M300 826 h840 a30 30 0 0 1 30 30 v336 a42 42 0 0 1 -42 42 h-816 a42 42 0 0 1 -42 -42 v-336 a30 30 0 0 1 30 -30z" fill="#e5dac1"/>
</g>
<path d="M300 826 h840 a30 30 0 0 1 30 30 v20 h-900 v-20 a30 30 0 0 1 30 -30z" fill="#f1e9d6"/>
<path d="M270 1100 h900 v92 a42 42 0 0 1 -42 42 h-816 a42 42 0 0 1 -42 -42z" fill="#d2c4a4"/>
<rect x="430" y="896" width="590" height="24" rx="12" fill="#3d362b"/>
<!-- handset -->
<g filter="url(#shs)">
  <rect x="410" y="702" width="330" height="54" rx="27" fill="#7b7464"/>
  <ellipse cx="438" cy="728" rx="62" ry="40" fill="#857d6b"/><ellipse cx="712" cy="728" rx="62" ry="40" fill="#857d6b"/>
  <ellipse cx="430" cy="716" rx="34" ry="14" fill="#fff" opacity=".18"/><ellipse cx="704" cy="716" rx="34" ry="14" fill="#fff" opacity=".18"/>
</g>
{keys}
<!-- lcd + buttons -->
<rect x="790" y="950" width="320" height="104" rx="14" fill="#5f5a4c"/>
<rect x="802" y="962" width="296" height="80" rx="8" fill="#9fb488"/>
<text x="950" y="996" text-anchor="middle" font-family="DejaVu Sans Mono" font-weight="700" font-size="23" fill="#2b3a20">LIGNE OCCUPÉE</text>
<text x="950" y="1028" text-anchor="middle" font-family="DejaVu Sans Mono" font-size="19" fill="#2b3a20">depuis 2009</text>
<circle cx="822" cy="1094" r="16" fill="#c9bc9c"/><circle cx="870" cy="1094" r="16" fill="#c9bc9c"/><circle cx="918" cy="1094" r="16" fill="#c9bc9c"/>
<rect x="972" y="1080" width="120" height="30" rx="15" fill="#3f8f4e"/><text x="1032" y="1102" text-anchor="middle" font-family="Inter" font-weight="800" font-size="17" fill="#fff">ENVOI</text>
<circle cx="1128" cy="972" r="12" fill="#ff3b30"/><circle cx="1128" cy="972" r="26" fill="#ff3b30" opacity=".25" filter="url(#blur6)"/>
{grill}
<text x="372" y="1176" font-family="Inter Display" font-weight="900" font-size="34" fill="#b4a37d" letter-spacing="3">FAXOMATIC 2000</text>
<rect x="270" y="826" width="900" height="408" rx="40" fill="url(#light)" opacity=".7"/>
<!-- cobweb + spider -->
<g stroke="#fff" stroke-width="2.4" fill="none" opacity=".9">{web}{rings}</g>
<line x1="1182" y1="440" x2="1182" y2="626" stroke="#fff" stroke-width="2"/>
{legs}
<ellipse cx="1182" cy="644" rx="17" ry="20" fill="#1a1a1a"/>
<circle cx="1176" cy="638" r="5" fill="#fff"/><circle cx="1189" cy="638" r="5" fill="#fff"/>
<circle cx="1177" cy="639" r="2.2" fill="#111"/><circle cx="1190" cy="639" r="2.2" fill="#111"/>
<!-- mug -->
<g transform="translate(-150 20)"><g filter="url(#shs)">
  <path d="M232 1070 h150 v150 a24 24 0 0 1 -24 24 h-102 a24 24 0 0 1 -24 -24z" fill="#fff"/>
  <path d="M382 1100 a40 40 0 0 1 0 90" stroke="#fff" stroke-width="18" fill="none"/>
  <ellipse cx="307" cy="1070" rx="75" ry="16" fill="#e8e2d6"/><ellipse cx="307" cy="1072" rx="64" ry="11" fill="#5b3a20"/>
</g>
<text font-family="Inter Display" font-weight="900" font-size="20" fill="#D7263D" text-anchor="middle"><tspan x="307" y="1132">MEILLEUR</tspan><tspan x="307" y="1158">CHEF</tspan><tspan x="307" y="1184">DU MONDE</tspan></text></g>
{specks([(240,860,4),(1250,770,3),(520,560,3),(1080,880,4),(170,980,3),(1230,1010,5),(820,520,3)])}
{stamp(540, 1010, -10)}
"""


# ---------------------------------------------------------------- PRINTED MAILS
def mails():
    import math
    sheets = []
    n = 52
    for i in range(n):
        y = 1222 - i * 13.4
        x = 560 + 46 * math.sin(i / 8.5) + i * 0.9
        rot = 3.2 * math.sin(i * 1.7)
        sheets.append(f'<g transform="translate({x:.1f} {y:.1f}) rotate({rot:.2f})">'
                      f'<path d="M-200 0 l40 -20 h360 l-40 20z" fill="#fdfcf9"/>'
                      f'<rect x="-200" y="0" width="360" height="9" fill="#efe9e3"/></g>')
        if i % 9 == 4:
            side = 1 if i % 2 else -1
            sheets.append(f'<g transform="translate({x + side*120:.1f} {y - 4:.1f}) rotate({side*14})">'
                          f'<rect x="-90" y="-6" width="190" height="10" rx="2" fill="#fff"/>'
                          f'<rect x="-60" y="-5" width="90" height="7" fill="#f6ff3b" opacity=".9"/></g>')
    top_y = 1222 - (n - 1) * 13.4
    top_x = 560 + 46 * math.sin((n - 1) / 8.5) + (n - 1) * .9

    def page(x, y, rot, scale=1.0, subject="RE: TR: RE: Point du lundi"):
        lines = "".join(f'<rect x="22" y="{96 + k*20}" width="{170 - (k*37) % 60}" height="7" rx="3" fill="#cfcac4"/>' for k in range(7))
        return f"""<g transform="translate({x} {y}) rotate({rot}) scale({scale})" filter="url(#shs)">
<rect width="220" height="290" rx="6" fill="#fff"/>
<text x="22" y="40" font-family="Inter" font-weight="700" font-size="15" fill="#333">De : Gérard</text>
<text x="22" y="64" font-family="Inter" font-size="13" fill="#555">Objet : {subject}</text>
<rect x="20" y="76" width="180" height="2" fill="#e3ded8"/>{lines}
<rect x="20" y="132" width="150" height="16" fill="#f6ff3b" opacity=".75"/>
<text x="150" y="268" font-family="Inter Display" font-weight="900" font-size="54" fill="#e8e3dc">@</text></g>"""

    puffs = "".join(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#8d8a88" opacity="{o}"/>' for cx, cy, r, o in
                    [(1060, 820, 44, .55), (1110, 780, 54, .5), (1040, 740, 40, .45), (1150, 720, 46, .4), (1090, 680, 38, .35), (1170, 650, 30, .3)])
    return DEFS + ground(700, 1240, 600) + f"""
{''.join(sheets)}
<g transform="translate({top_x:.0f} {top_y:.0f})"><path d="M-200 0 l40 -20 h360 l-40 20z" fill="#fff"/>
<rect x="-130" y="-14" width="120" height="7" fill="#f6ff3b" transform="skewX(-60)"/></g>
{page(140, 560, -14, .92)}
{page(250, 840, 9, .8, "Fwd: Fwd: URGENT")}
{page(1180, 520, 16, .62, "RE: (sans objet)")}
<!-- printer -->
<g filter="url(#sh)">
  <path d="M880 900 h360 a26 26 0 0 1 26 26 v250 a30 30 0 0 1 -30 30 h-352 a30 30 0 0 1 -30 -30 v-250 a26 26 0 0 1 26 -26z" fill="#e9e6e3"/>
  <path d="M912 900 l28 -60 h240 l28 60z" fill="#d6d2ce"/>
</g>
<path d="M942 846 l14 -70 h212 l12 70z" fill="#fff"/>
<rect x="854" y="900" width="412" height="40" fill="#f5f3f1"/>
<rect x="920" y="1030" width="280" height="22" rx="11" fill="#3b3836"/>
<path d="M928 1046 h264 l26 88 h-316z" fill="#fff" filter="url(#shs)"/>
<rect x="1100" y="960" width="130" height="44" rx="8" fill="#2d2b2a"/>
<text x="1165" y="988" text-anchor="middle" font-family="DejaVu Sans Mono" font-weight="700" font-size="16" fill="#ff4d3d">BOURRAGE</text>
<circle cx="900" cy="980" r="10" fill="#ff3b30"/>
<rect x="854" y="900" width="412" height="336" rx="30" fill="url(#light)" opacity=".6"/>
{puffs}
<!-- highlighter -->
<g transform="translate(150 1180) rotate(-18)" filter="url(#shs)">
  <rect width="250" height="54" rx="14" fill="#f2f23a"/><rect x="190" width="60" height="54" rx="14" fill="#dada1e"/>
  <path d="M250 12 l40 10 v10 l-40 10z" fill="#c9c90f"/><rect x="20" y="10" width="150" height="10" rx="5" fill="#fff" opacity=".45"/>
  <text x="96" y="40" text-anchor="middle" font-family="Inter Display" font-weight="900" font-size="20" fill="#7c7c00">STABILO</text>
</g>
{specks([(820,600,4),(1000,560,3),(380,520,3),(1280,880,4),(160,1020,3)])}
{stamp(560, 1120, -8, "100% BOOMER**", size=34)}
"""


# ---------------------------------------------------------------- EXCEL
def excel():
    cols = "ABCDEFG"
    cw, rh, x0, y0 = 152, 46, 64, 196
    head = "".join(f'<rect x="{x0 + i*cw}" y="{y0}" width="{cw}" height="34" fill="#eef0ee" stroke="#d3d6d3"/>'
                   f'<text x="{x0 + i*cw + cw/2}" y="{y0+24}" text-anchor="middle" font-family="Inter" font-size="17" fill="#555">{c}</text>' for i, c in enumerate(cols))
    data = [
        ["Nom", "Tél", "Statut", "Relance", "Notes", "CA ?", ""],
        ["M. Durand", "01 42 ...", "à rappeler ???", "#REF!", "VOIR AVEC GÉRARD", "#N/A", ""],
        ["Durand M.", "01 42 ...", "doublon ?", "2017", "", "#DIV/0!", ""],
        ["SARL Martin", "fax only", "chaud 🔥", "jamais", "il a dit peut-être", "####", ""],
        ["Mme Petit", "", "froid", "#REF!", "NE PAS EFFACER", "12", ""],
        ["???", "voir post-it", "", "", "", "#VALEUR!", ""],
        ["M. Durand", "01 42 ...", "re-doublon", "#REF!", "", "", ""],
        ["", "", "", "", "", "", ""],
    ]
    cells = []
    for r, row in enumerate(data):
        y = y0 + 34 + r * rh
        cells.append(f'<rect x="20" y="{y}" width="44" height="{rh}" fill="#eef0ee" stroke="#d3d6d3"/>'
                     f'<text x="42" y="{y+30}" text-anchor="middle" font-family="Inter" font-size="16" fill="#666">{r+1}</text>')
        for c, v in enumerate(row):
            x = x0 + c * cw
            fill = "#fff"
            if v.startswith("#"):
                fill = "#ffd9d6"
            elif v in ("à rappeler ???", "NE PAS EFFACER", "chaud 🔥", "VOIR AVEC GÉRARD"):
                fill = "#f6ff3b"
            if r == 0:
                fill = "#d8ecd9"
            cells.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{rh}" fill="{fill}" stroke="#dfe2df"/>')
            if v:
                weight = 700 if r == 0 or v.startswith("#") else 400
                color = "#c0201a" if v.startswith("#") else "#222"
                cells.append(f'<text x="{x+10}" y="{y+30}" font-family="Inter" font-size="16.5" font-weight="{weight}" fill="{color}">{v}</text>')
    tabs = ["Feuil1", "Feuil2", "Feuil3 (2)", "NE PAS TOUCHER", "copie de copie", "Feuil47"]
    tx, tab_svg = 20, []
    for i, t in enumerate(tabs):
        w = len(t) * 10.5 + 34
        active = i == 3
        tab_svg.append(f'<rect x="{tx}" y="640" width="{w}" height="36" rx="4" fill="{"#fff" if active else "#e3e6e3"}" stroke="#cfd3cf"/>'
                       f'<text x="{tx + w/2}" y="664" text-anchor="middle" font-family="Inter" font-size="15" font-weight="{700 if active else 400}" fill="{"#1D6F42" if active else "#555"}">{t}</text>')
        tx += w + 6
    ribbon = "".join(f'<rect x="{30 + i*58}" y="108" width="40" height="30" rx="6" fill="#dfe5df"/>' for i in range(16))
    return DEFS + ground(720, 1250, 600) + f"""
<g transform="translate(150 478) rotate(-3.5)">
  <g filter="url(#sh)"><rect width="1160" height="700" rx="24" fill="#fff"/></g>
  <path d="M0 24 a24 24 0 0 1 24 -24 h1112 a24 24 0 0 1 24 24 v40 h-1160z" fill="#1D6F42"/>
  <circle cx="34" cy="32" r="9" fill="#ff5f57"/><circle cx="62" cy="32" r="9" fill="#febc2e"/><circle cx="90" cy="32" r="9" fill="#28c840"/>
  <text x="580" y="42" text-anchor="middle" font-family="Inter" font-weight="700" font-size="22" fill="#fff">Clients_FINAL_v12_OK_vraimentfinal(2).xls</text>
  <rect y="64" width="1160" height="90" fill="#f5f7f5"/>{ribbon}
  <rect x="20" y="152" width="1120" height="36" rx="4" fill="#fff" stroke="#d3d6d3"/>
  <text x="34" y="177" font-family="DejaVu Sans Mono" font-size="16" fill="#333"><tspan font-style="italic" fill="#1D6F42" font-weight="700">fx</tspan>  =SI(RECHERCHEV(A2;Feuil47!B:B;3;FAUX)="";"demander à Gérard";#REF!)</text>
  {head}{''.join(cells)}
  <rect x="0" y="632" width="1160" height="68" fill="#f5f7f5"/>{''.join(tab_svg)}
  <path d="M0 676 h1160 v0 a24 24 0 0 1 -24 24 h-1112 a24 24 0 0 1 -24 -24z" fill="#1D6F42"/>
</g>
<!-- post-it -->
<g transform="translate(96 560) rotate(-9)" filter="url(#shs)">
  <rect width="210" height="200" fill="#ffe55c"/><rect width="210" height="34" fill="#f5d63c"/>
  <text font-family="Gochi" font-size="33" fill="#1e2a78"><tspan x="16" y="78">NE PAS</tspan><tspan x="16" y="116">supprimer</tspan><tspan x="16" y="154">colonne F !!!</tspan></text>
</g>
<!-- frozen dialog -->
<g transform="translate(560 1000) rotate(3)" filter="url(#sh)">
  <rect width="470" height="200" rx="18" fill="#fff"/>
  <circle cx="58" cy="70" r="30" fill="#f2c94c"/><text x="58" y="84" text-anchor="middle" font-family="Inter Display" font-weight="900" font-size="38" fill="#fff">!</text>
  <text font-family="Inter" font-size="21" fill="#222"><tspan x="108" y="60" font-weight="700">Excel ne répond pas.</tspan><tspan x="108" y="92">Voulez-vous attendre ?</tspan></text>
  <rect x="108" y="128" width="148" height="46" rx="10" fill="#1D6F42"/><text x="182" y="158" text-anchor="middle" font-family="Inter" font-weight="700" font-size="19" fill="#fff">Attendre</text>
  <rect x="270" y="128" width="168" height="46" rx="10" fill="#eceeec"/><text x="354" y="158" text-anchor="middle" font-family="Inter" font-weight="700" font-size="19" fill="#333">Pleurer</text>
</g>
<!-- hourglass cursor -->
<g transform="translate(1150 760) rotate(12)" filter="url(#shs)">
  <path d="M0 0 h80 v12 c0 30 -30 44 -30 58 c0 14 30 28 30 58 v12 h-80 v-12 c0 -30 30 -44 30 -58 c0 -14 -30 -28 -30 -58z" fill="#fff" stroke="#222" stroke-width="5"/>
  <path d="M16 18 h48 c-6 22 -24 30 -24 40 c0 -10 -18 -18 -24 -40z M40 100 c-14 4 -22 18 -24 30 h48 c-2 -12 -10 -26 -24 -30z" fill="#f2c94c"/>
</g>
{stamp(1060, 560, 8, "100% BOOMER**", size=34)}
{specks([(1300,700,4),(110,900,3),(640,450,3)])}
"""


# ---------------------------------------------------------------- ANSWERING MACHINE
def repondeur():
    cord = "M470 1150 " + " ".join(f"c 10 -26 34 -26 30 0" for _ in range(1))
    coil = "".join(f'<ellipse cx="{470 + i*17}" cy="{1196 + 10*((i%2)*2-1)*0}" rx="11" ry="24" fill="none" stroke="#b9372c" stroke-width="7"/>' for i in range(14))
    bubbles = [
        (120, 500, -3, ["Oui bonjour, c'est Jean-Claude,", "rappelez-moi !"], 610),
        (380, 662, 2, ["C'est encore Jean-Claude.", "Vous avez eu mon fax ?"], 590),
        (180, 818, -2, ["JEAN-CLAUDE. RAPPELEZ-MOI."], 610),
    ]
    bsvg = ""
    for x, y, rot, lines, w in bubbles:
        h = 34 + len(lines) * 42
        txt = "".join(f'<tspan x="34" y="{52 + k*42}">{l}</tspan>' for k, l in enumerate(lines))
        bsvg += f"""<g transform="translate({x} {y}) rotate({rot})" filter="url(#shs)">
<rect width="{w}" height="{h}" rx="30" fill="#fff"/><path d="M{w-170} {h-2} l40 44 l4 -44z" fill="#fff"/>
<text font-family="Inter Display" font-weight="700" font-size="32" fill="#1b1b1b">{txt}</text></g>"""
    postits = [(1150, 560, 8, "#ffe55c", ["Rappeler", "J-C"]), (1210, 760, -6, "#ff9fc4", ["Rappeler", "J-C !!"])]
    psvg = "".join(
        f'<g transform="translate({x} {y}) rotate({r})" filter="url(#shs)"><rect width="160" height="150" fill="{c}"/>'
        f'<text font-family="Gochi" font-size="34" fill="#1e2a78">' + "".join(f'<tspan x="16" y="{58 + k*40}">{t}</tspan>' for k, t in enumerate(ls)) + "</text></g>"
        for x, y, r, c, ls in postits)
    return DEFS + ground(760, 1236, 560) + f"""
{bsvg}{psvg}
<!-- answering machine -->
<g filter="url(#sh)">
  <path d="M600 960 h560 q24 0 30 20 l24 70 h-668 l24 -70 q6 -20 30 -20z" fill="#3a3a3e"/>
  <path d="M556 1050 h668 a20 20 0 0 1 20 20 v130 a34 34 0 0 1 -34 34 h-640 a34 34 0 0 1 -34 -34 v-130 a20 20 0 0 1 20 -20z" fill="#232326"/>
</g>
<rect x="640" y="976" width="300" height="64" rx="10" fill="#151517" transform="skewX(-14) translate(240 0)"/>
<g transform="translate(700 984)"><rect width="250" height="50" rx="8" fill="#6d5b4a"/>
  <circle cx="72" cy="25" r="17" fill="#eee"/><circle cx="178" cy="25" r="17" fill="#eee"/>
  <circle cx="72" cy="25" r="6" fill="#333"/><circle cx="178" cy="25" r="6" fill="#333"/></g>
<rect x="600" y="1090" width="240" height="104" rx="12" fill="#0b0b0c"/>
<text x="720" y="1170" text-anchor="middle" font-family="DejaVu Sans Mono" font-weight="700" font-size="76" fill="#ff3b30">99+</text>
<text x="720" y="1170" text-anchor="middle" font-family="DejaVu Sans Mono" font-weight="700" font-size="76" fill="#ff3b30" filter="url(#blur6)" opacity=".7">99+</text>
<g font-family="Inter" font-weight="800" font-size="18" fill="#cfcfd4" text-anchor="middle">
  <rect x="880" y="1096" width="130" height="44" rx="12" fill="#3c3c41"/><text x="945" y="1125">▶ PLAY</text>
  <rect x="1030" y="1096" width="150" height="44" rx="12" fill="#3c3c41"/><text x="1105" y="1125">EFFACER</text>
  <rect x="880" y="1150" width="300" height="44" rx="12" fill="#3c3c41"/><text x="1030" y="1179">MESSAGES : TOUS DE J-C</text>
</g>
<circle cx="1196" cy="1080" r="9" fill="#ff3b30"/><circle cx="1196" cy="1080" r="22" fill="#ff3b30" opacity=".35" filter="url(#blur6)"/>
<!-- desk phone -->
<g filter="url(#sh)">
  <path d="M250 1010 h230 q30 0 40 28 l50 150 q8 46 -40 46 h-330 q-48 0 -40 -46 l50 -150 q10 -28 40 -28z" fill="#d84a3c"/>
</g>
<g filter="url(#shs)">
  <path d="M176 980 q-20 -60 50 -60 h300 q70 0 50 60 q-10 30 -46 26 l-40 -4 q-20 -2 -26 -22 l-6 -14 h-186 l-6 14 q-6 20 -26 22 l-40 4 q-36 4 -24 -26z" fill="#e5594a"/>
</g>
<g fill="#f7efe8">{''.join(f'<rect x="{306 + c*46}" y="{1066 + r*38}" width="36" height="28" rx="8"/>' for r in range(3) for c in range(3))}</g>
<path d="M190 1000 q-40 60 -10 140 q20 50 60 70" stroke="#b9372c" stroke-width="10" fill="none" stroke-dasharray="4 6" stroke-linecap="round"/>
<path d="M250 1010 h230 q30 0 40 28 l50 150 q8 46 -40 46 h-330 q-48 0 -40 -46 l50 -150 q10 -28 40 -28z" fill="url(#light)" opacity=".7"/>
{stamp(1000, 900, -7, "100% BOOMER**", size=32)}
{specks([(560,920,4),(1290,1000,3),(60,880,3)])}
"""


# ---------------------------------------------------------------- MINITEL
def minitel():
    return DEFS + ground(700, 1244, 560) + f"""
<!-- yellow pages -->
<g filter="url(#sh)">
  <g transform="translate(140 1006) rotate(-4)"><rect width="330" height="90" rx="10" fill="#f3c500"/><rect y="70" width="330" height="20" fill="#f6f1e1"/>
  <text x="24" y="52" font-family="Inter Display" font-weight="900" font-size="34" fill="#111">PAGES JAUNES</text></g>
  <g transform="translate(120 1104) rotate(2)"><rect width="370" height="110" rx="10" fill="#f3c500"/><rect y="86" width="370" height="24" fill="#f6f1e1"/>
  <text x="24" y="60" font-family="Inter Display" font-weight="900" font-size="36" fill="#111">PAGES JAUNES</text>
  <text x="318" y="60" font-family="Inter Display" font-weight="700" font-size="22" fill="#111">1994</text></g>
</g>
<!-- minitel body -->
<g filter="url(#sh)">
  <path d="M600 520 h520 a40 40 0 0 1 40 40 v420 a30 30 0 0 1 -30 30 h-540 a30 30 0 0 1 -30 -30 v-420 a40 40 0 0 1 40 -40z" fill="#8a6b4b"/>
  <path d="M520 1010 h680 l60 160 a30 30 0 0 1 -28 40 h-744 a30 30 0 0 1 -28 -40z" fill="#7a5d40"/>
</g>
<path d="M600 520 h520 a40 40 0 0 1 40 40 v420 a30 30 0 0 1 -30 30 h-540 a30 30 0 0 1 -30 -30 v-420 a40 40 0 0 1 40 -40z" fill="url(#light)" opacity=".6"/>
<rect x="612" y="560" width="496" height="390" rx="46" fill="#3a2c20"/>
<rect x="632" y="578" width="456" height="352" rx="38" fill="#0f1a14"/>
<g font-family="DejaVu Sans Mono" font-weight="700">
  <rect x="668" y="610" width="384" height="64" fill="#2b6cd9"/>
  <text x="860" y="656" text-anchor="middle" font-size="40" fill="#fff">3615 PROSPECT</text>
  <text x="672" y="724" font-size="22" fill="#5cf2ff">Nom : ______________</text>
  <text x="672" y="760" font-size="22" fill="#5cf2ff">Ville : ____________</text>
  <text x="672" y="808" font-size="22" fill="#ffe14d">Recherche en cours...</text>
  <text x="672" y="844" font-size="22" fill="#ffe14d">▓▓▓▓░░░░░░░ 12 %</text>
  <text x="672" y="900" font-size="18" fill="#ff7ad9">0,34 F/min · ENVOI</text>
</g>
<rect x="632" y="578" width="456" height="352" rx="38" fill="url(#light)" opacity=".35"/>
<text x="860" y="990" text-anchor="middle" font-family="Inter Display" font-weight="900" font-size="22" fill="#d8c3a5" letter-spacing="6">MINITEL</text>
<g font-family="Inter" font-weight="800" font-size="15" fill="#3d2c1c" text-anchor="middle">
{''.join(f'<rect x="{574 + c*48 + r*8}" y="{1040 + r*40}" width="40" height="30" rx="7" fill="#e9dcc7"/>' for r in range(3) for c in range(12))}
  <rect x="610" y="1162" width="120" height="30" rx="7" fill="#3f8f4e"/><text x="670" y="1183" fill="#fff">ENVOI</text>
  <rect x="742" y="1162" width="120" height="30" rx="7" fill="#e9dcc7"/><text x="802" y="1183">SUITE</text>
  <rect x="874" y="1162" width="120" height="30" rx="7" fill="#e9dcc7"/><text x="934" y="1183">RETOUR</text>
  <rect x="1006" y="1162" width="150" height="30" rx="7" fill="#d84a3c"/><text x="1081" y="1183" fill="#fff">CONNEXION</text>
</g>
{stamp(340, 820, -9, "100% BOOMER**", size=34)}
{specks([(1250,640,4),(520,600,3),(1300,880,3),(260,620,4)])}
"""

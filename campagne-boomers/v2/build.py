import base64, pathlib
H = pathlib.Path(__file__).parent
gochi = base64.b64encode((H.parent / 'assets/GochiHand.woff2').read_bytes()).decode()
RED = "#E8412F"
POSTERS = {
    "metro-worker-dodo-kate": ("kate", "boulot"),
    "metro-worker-dodo-sophie": ("sophie", "prospection"),
    "metro-worker-dodo-james": ("james", "mails"),
}
for name, (base, old) in POSTERS.items():
    img = base64.b64encode((H / f'base/{base}.png').read_bytes()).decode()
    (H / f'{name}.html').write_text(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Métro worker dodo</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{gochi}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:url(data:image/png;base64,{img}) 0 0/2000px 1500px; }}
h1 {{ position:absolute; left:86px; top:52px; font-size:116px; line-height:1.0; font-weight:500; letter-spacing:-3px; color:#000; }}
h1 b {{ font-weight:800; }}
h1 sup {{ font-size:56px; font-weight:500; vertical-align:52px; margin-left:3px; }}
.old {{ position:relative; display:inline-block; }}
.old svg {{ position:absolute; left:-12px; top:0; width:calc(100% + 24px); height:100%; overflow:visible; }}
.note {{ position:absolute; left:calc(100% + 26px); top:22px; font-family:Gochi; font-size:46px; letter-spacing:0; color:{RED};
        transform:rotate(-6deg); white-space:nowrap; font-weight:400; }}
</style></head><body><div class="p">
<h1>Métro,<br><span class="old">{old}<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<path d="M0 58 C 20 46, 40 64, 60 50 S 90 44, 100 54" stroke="{RED}" stroke-width="8" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
<path d="M3 66 C 25 54, 45 72, 70 58 S 92 52, 98 62" stroke="{RED}" stroke-width="5" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>
<span class="note">← méthode boomer</span></span><br><b>worker<sup>*</sup>,</b><br>dodo.</h1>
</div></body></html>""")
print('ok')

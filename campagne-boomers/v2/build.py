import base64, pathlib
H = pathlib.Path(__file__).parent
gochi = base64.b64encode((H.parent / 'assets/GochiHand.woff2').read_bytes()).decode()
POSTERS = {
    "eat-sleep-worker-sophie": ("sophie", "Prospecter, relancer, re-relancer.", "#E8412F"),
    "eat-sleep-worker-james": ("james", "Mails, réunions, re-mails.", "#E8412F"),
    "eat-sleep-worker-kate": ("kate", "Métro, boulot, dodo.", "#E8412F"),
}
for name, (base, old, red) in POSTERS.items():
    img = base64.b64encode((H / f'base/{base}.png').read_bytes()).decode()
    (H / f'{name}.html').write_text(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Eat sleep worker repeat</title><style>
@font-face {{ font-family: Gochi; src: url(data:font/woff2;base64,{gochi}); }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:2000px; height:1500px; overflow:hidden; font-family:'Inter Display'; -webkit-font-smoothing:antialiased; }}
.p {{ position:relative; width:2000px; height:1500px; background:url(data:image/png;base64,{img}) 0 0/2000px 1500px; }}
.old {{ position:absolute; left:90px; top:70px; font-size:84px; font-weight:500; letter-spacing:-1.5px; color:#111; }}
.old svg {{ position:absolute; left:-14px; top:6px; width:calc(100% + 28px); height:100px; overflow:visible; }}
.note {{ position:absolute; top:58px; font-family:Gochi; font-size:46px; color:{red}; transform:rotate(-6deg); white-space:nowrap; }}
h1 {{ position:absolute; left:84px; top:184px; font-size:166px; line-height:1.0; font-weight:800; letter-spacing:-4px; color:#000; }}
h1 .w {{ position:relative; }}
h1 sup {{ font-size:76px; font-weight:500; vertical-align:74px; margin-left:4px; }}
</style></head><body><div class="p">
<div class="old" id="old">{old}<svg viewBox="0 0 100 100" preserveAspectRatio="none">
<path d="M0 58 C 20 46, 40 62, 60 50 S 90 44, 100 52" stroke="{red}" stroke-width="7" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
<path d="M2 64 C 25 54, 45 70, 70 56 S 92 52, 99 60" stroke="{red}" stroke-width="5" fill="none" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg></div>
<div class="note" id="note">← méthode boomer</div>
<h1>Eat, sleep,<br><span class="w">worker<sup>*</sup></span>, repeat.</h1>
</div><script>
const o=document.getElementById('old'), n=document.getElementById('note');
n.style.left=(o.offsetLeft+o.offsetWidth+30)+'px';
</script></body></html>""")
print('ok')

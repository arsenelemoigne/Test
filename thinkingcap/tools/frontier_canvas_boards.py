"""Rebuild the Types and Heatmap boards of the Frontier canvas from Main.dc.html's DATA (full tests only)."""
import json, sys, glob, html
P = glob.glob(sys.argv[1])[0]
s = open(f'{P}/Main.dc.html').read(); i = s.index('const DATA = ') + 13
D, _ = json.JSONDecoder().raw_decode(s[i:])
M = {m['key']: m for m in D['models']}
TY = [('v', 'Math'), ('c', 'Code'), ('f', 'Instructions (IFEval)'), ('s', 'Tools · one function'), ('m', 'Tools · several'), ('i', 'Tools · none fits')]
INK, MU, ORANGE, BLUE = '#1B1D22', '#585D66', '#C2531F', '#2a78d6'
def sa(m, k):  # (saved, acc) of a model on type k, or overall
    if k == 'all': return m['saved'], m['acc']
    p = m['parts'][k]; return (p['saved'], p['acc']) if isinstance(p, dict) else (p[0], p[1])
def ref_sa(r, k):
    if k == 'all': return r['saved'], r['acc']
    return (r['parts'][k][0], r['parts'][k][1]) if r['parts'] else None
SHOW = ['rpfull', 'rpkfull', 'rplfull', 'rpmfull', 'path1r2', 'path2od5c', 'path2od5r', 'path2od5f']
SHORT = {'rpfull': 'rc1 → rpE', 'rpkfull': 'rpK', 'rplfull': 'rpL', 'rpmfull': 'rpM', 'path1r2': 'Path 1 · offline DPO', 'path2od5c': 'online DPO od5c',
         'path2od5r': 'od5c, second seed (od5r)', 'path2od5f': 'od5f (od5c tuned)'}
FAM = [f for k in SHOW for f in D['fams'] if f['key'] == k]
lab = lambda key: key.replace('full', '').replace('-r', ' r').replace('rpe', 'rpE').replace('rpk', 'rpK').replace('rpl', 'rpL').replace('rpm', 'rpM').replace('rpi', 'rpI')
HEAD = s[:0]
T = open(f'{P}/Types.dc.html').read(); H = open(f'{P}/Heatmap.dc.html').read()

def panel(k, title):
    ok = lambda key: key in M and (k == 'all' or M[key].get('parts'))
    pts = [sa(M[key], k) for f in FAM for key in f['keys'] if ok(key)]
    refs = [(r['name'], ref_sa(r, k)) for r in D['refs'] if ref_sa(r, k)]
    xs_ = [p[0] for p in pts] + [p[0] for _, p in refs]; ys_ = [p[1] for p in pts] + [p[1] for _, p in refs] + [0]
    x0, x1 = min(xs_) - 3, max(xs_) + 3; y0, y1 = min(ys_) - 0.8, max(ys_) + 0.8
    if k == 'all': x0, x1, y0, y1 = min(x0, 38), max(x1, 66), min(y0, -4.2), max(y1, 0.8)
    W, Hh, L, R, Tp, B = 440, 330, 44, 14, 14, 44
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R); Y = lambda v: Tp + (y1 - v) / (y1 - y0) * (Hh - Tp - B)
    g = [f'<svg viewBox="0 0 {W} {Hh}" style="width: 100%; height: auto" role="img" aria-label="Full tests, {html.escape(title)}">']
    if k == 'all':
        g.append(f'<rect x="{X(50):.0f}" y="{Tp}" width="{W - R - X(50):.0f}" height="{Y(-1) - Tp:.0f}" fill="#E4EEFA" stroke="{BLUE}" stroke-dasharray="4 3"/>')
        g.append(f'<text x="{W - R - 4}" y="{Tp + 13}" text-anchor="end" font-size="10.5" fill="{BLUE}">goal: &gt; 50% saved, better than −1</text>')
    import math
    step = 5 if x1 - x0 < 45 else 10
    for v in range(int(math.ceil(x0 / step) * step), int(x1) + 1, step):
        g.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Tp}" y2="{Hh - B}" stroke="#EEF0EC"/><text x="{X(v):.1f}" y="{Hh - B + 16}" text-anchor="middle" font-size="11" fill="{MU}">{v}%</text>')
    ystep = 1 if y1 - y0 <= 8 else 2 if y1 - y0 <= 16 else 5
    for v in range(int(math.ceil(y0 / ystep) * ystep), int(math.floor(y1)) + 1, ystep):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{"#9AA0A8" if v == 0 else "#EEF0EC"}"/><text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="{MU}">{"+" if v > 0 else "−" if v < 0 else ""}{abs(v)}</text>')
    g.append(f'<text x="{(L + W - R) / 2}" y="{Hh - 8}" text-anchor="middle" font-size="11.5" fill="{MU}">tokens saved →</text>')
    for name, (x, y) in refs:
        g.append(f'<rect x="{X(x) - 5:.1f}" y="{Y(y) - 5:.1f}" width="10" height="10" fill="#FFFFFF" stroke="{INK}" stroke-width="1.6"><title>{html.escape(name)}: {x:.1f}% / {y:+.2f}</title></rect>')
    labels = []
    for f in FAM:
        keys = [key for key in f['keys'] if ok(key)]; P2 = [sa(M[key], k) for key in keys]
        hot = f['key'] == 'path2od5f'
        g.append(f'<polyline points="{" ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in P2)}" fill="none" stroke="{f["color"]}" stroke-width="{2.8 if hot else 1.8}" stroke-linejoin="round" opacity="{1 if hot else .8}"/>')
        for key, (x, y) in zip(keys, P2):
            g.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{5.5 if hot else 4.2}" fill="{f["color"]}" stroke="#FFFFFF" stroke-width="1.2"><title>{html.escape(M[key]["name"])}: {x:.1f}% / {y:+.2f}</title></circle>')
        if hot or f['key'] in ('path2od5c', 'rpmfull', 'rplfull'):
            key = 'od5f-r18full' if hot else 'od5c-r16full' if f['key'] == 'path2od5c' else keys[-1]
            if ok(key): x, y = sa(M[key], k); labels.append((X(x), Y(y) + (24 if hot else 0), lab(key), f['color']))
    for x, y, t, c in labels:
        anchor = 'end' if x > W - 90 else 'start'
        g.append(f'<text x="{x + (-8 if anchor == "end" else 8):.1f}" y="{y - 7:.1f}" text-anchor="{anchor}" fill="{c}" font-size="11" font-weight="600" paint-order="stroke" stroke="#FFFFFF" stroke-width="3">{t}</text>')
    g.append('</svg>')
    return f'<div style="display: flex; flex-direction: column; gap: 4px; background: #FFFFFF; border: 1px solid #E1E4DF; border-radius: 8px; padding: 14px"><div style="font-weight: 600; font-size: 15px">{title}</div>' + '\n'.join(g) + '</div>'

# ---- Types board
legend = '<div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 14px; color: #585D66">' + ''.join(
    f'<span style="display: flex; gap: 7px; align-items: center"><span style="width: 22px; height: {5 if f["key"] == "path2od5f" else 4}px; border-radius: 2px; background: {f["color"]}"></span>{SHORT[f["key"]]}</span>' for f in FAM) + \
    f'<span style="display: flex; gap: 7px; align-items: center"><span style="width: 10px; height: 10px; border: 1.6px solid {INK}; background: #FFFFFF"></span>other teams</span></div>'
how = f'<div style="display: flex; flex-direction: column; gap: 8px; padding: 14px; font-size: 13.5px; color: {MU}"><b style="color: {INK}">How to read</b><span>Every full test of each branch, by type. Each line moves through its checkpoints in training order.</span><span>od5f (thick line) is od5c tuned: rounds 14 to 18. Round 18 saves the most inside the goal (59.5% / −0.80); the extra saving comes mostly from math (71% saved, −3.6).</span><span>Some teams report only an overall score, so they appear only in the first panel.</span></div>'
grid = '<div style="display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px">' + panel('all', 'All types (official weighting)') + ''.join(panel(k, t) for k, t in TY) + how + '</div>'
a = T.index('<div style="display: flex; gap: 20px; flex-wrap: wrap'); b = T.index('</x-dc>')
T = T[:a] + legend + grid + '\n</div>\n' + T[b:]
T = T.replace('The full-test checkpoints of our repair branches, drawn on each question type separately.', 'The full-test checkpoints of each branch, offline DPO and online DPO, drawn on each question type separately.')
T = T.replace('height: 1060px', 'height: 1120px').replace('"height": 1060', '"height": 1120')
open(f'{P}/Types.dc.html', 'w').write(T)

# ---- Heatmap board
rows = []
for m in D['models']:
    if m['kind'] == 'full' and m.get('parts'): rows.append((m['saved'], m['acc'], lab(m['key']) if not m['key'].startswith(('pt-', 'bida', 'clean')) else m['name'].split(' · ')[0] + (' · ' + m['name'].split(' · ')[1] if ' · ' in m['name'] else ''), [sa(m, k) for k, _ in TY], m['key'] == 'od5f-r18full', False))
for r in D['refs']:
    if r['parts']: rows.append((r['saved'], r['acc'], r['name'] + ' (other team)', [ref_sa(r, k) for k, _ in TY], False, True))
o = D['off']; rows.append((o['saved'], o['acc'], 'Original, thinking off', [tuple(o['parts'][k]) for k, _ in TY], False, True))
rows.sort(key=lambda r: -r[0])
def cell(x, y):
    a_ = min(0.93, abs(y) / 5 * 0.93 + 0.05); col = '194,83,31' if y < 0 else '42,120,214'
    return f'<td style="padding: 3px"><div style="background: rgba({col},{a_:.2f}); border-radius: 4px; padding: 6px 4px; text-align: center; color: {"#FFFFFF" if a_ > 0.6 else INK}"><div style="font-weight: 600">{"+" if y > 0 else "−" if y < 0 else "±"}{abs(y):.1f}</div><div style="font-size: 11px; opacity: 0.8">{x:.0f}% saved</div></div></td>'
body = ''
for sv, ac, name, parts, best, ital in rows:
    inside = sv > 50 and ac > -1
    st = 'font-style: italic; color: #585D66; ' if ital else ('font-weight: 700; color: #C21F5B; ' if best else '')
    body += (f'<tr{" style=\"background: #FBEFF3\"" if best else ""}><td style="padding: 6px 10px 6px 4px; white-space: nowrap; font-family: \'IBM Plex Sans\', sans-serif; color: {INK}; {st}">{html.escape(name)}{" ★" if best else ""}</td>'
             f'<td style="padding: 6px 8px; text-align: right; font-weight: 600">{sv:.1f}%</td><td style="padding: 6px 8px; text-align: right; font-weight: 600; color: {ORANGE if ac < 0 else BLUE}">{"−" if ac < 0 else "+"}{abs(ac):.2f}</td>'
             f'<td style="padding: 6px 8px; font-size: 11px; color: {BLUE}">{"in goal" if inside and not ital else ""}</td>' + ''.join(cell(x, y) for x, y in parts) + '</tr>')
a = H.index('<tbody>') + 7; b = H.index('</tbody>')
H = H[:a] + body + H[b:]
H = H.replace('Rows sorted by tokens saved.', 'Rows sorted by tokens saved; ★ = od5f round 18, the most saving inside the goal.')
hh = 340 + 56 * len(rows)
import re
H = re.sub(r'height: \d+px; box-sizing', f'height: {hh}px; box-sizing', H, count=1); H = re.sub(r'"height": \d+', f'"height": {hh}', H)
open(f'{P}/Heatmap.dc.html', 'w').write(H)
c = json.load(open(f'{P}/canvas.json')); c['boards']['Heatmap.dc.html']['h'] = hh + 20; c['boards']['Types.dc.html']['h'] = 1120
json.dump(c, open(f'{P}/canvas.json', 'w'), indent=2)
print('rows', len(rows), 'heat h', hh, 'fams', [f['key'] for f in FAM])

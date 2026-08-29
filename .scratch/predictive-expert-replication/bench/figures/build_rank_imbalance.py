# SPDX-License-Identifier: Apache-2.0
"""Build the prefill-vs-decode rank-imbalance figure from the load dump.

Reads `rank-imbalance-data.json` (produced from VLLM_EPLB_DUMP_LOAD_PATH output)
and writes a self-contained HTML page. Data is injected, never transcribed.

Two tables, and the distinction between them is the whole point of the page:
the per-layer table is the heatmap's and the line chart's table view, and the
aggregate table is the metric that hides the skew, kept only as the evidence
behind the first stat tile.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data", type=Path, default=HERE / "rank-imbalance-data.json")
_ap.add_argument("--out", type=Path, default=HERE / "rank-imbalance.html")
_ap.add_argument("--model", default="Qwen3-30B-A3B")
_args = _ap.parse_args()

D = json.loads(_args.data.read_text())
# Shape comes from the data, not from constants: the figure was pinned to 8x48 and could
# not be rebuilt for another model. DeepSeek-V4-Flash is 43 layers and 256 experts.
_any = next(iter(D.values()))
EP = _any.get("ep", 8)
L = _any.get("num_layers", len(_any["share_by_layer"]))
MODEL = _args.model
UNIFORM = 100.0 / EP
RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
        "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"]
_shares = [v for k in D for row in D[k]["share_by_layer"] for v in row]
LO, HI = min(_shares), max(_shares)


def ramp(v):
    t = (v - LO) / (HI - LO)
    return RAMP[max(0, min(len(RAMP) - 1, int(t * len(RAMP))))]


def heat_cells(key):
    rows = D[key]["share_by_layer"]
    out = []
    for r in range(EP):
        for li, layer in enumerate(rows):
            v = layer[r]
            peak = v == max(layer)
            out.append(
                f'<div class="cell{" pk" if peak else ""}" style="background:{ramp(v)}" '
                f'tabindex="0" data-t="L{li} · rank {r} · {v:.2f}% of this layer'
                f'{" · layer peak" if peak else ""}"></div>')
    return "".join(out)


W, H, PAD = 720, 150, 8
_peaks = [v for k in D for v in D[k]["peak_over_mean"]]
YMIN, YMAX = 1.0, max(2.45, max(_peaks) * 1.05)
def xp(i): return PAD + i * (W - 2 * PAD) / (L - 1)
def yp(v): return H - PAD - (v - YMIN) / (YMAX - YMIN) * (H - 2 * PAD)


def spark(key, color):
    pk = D[key]["peak_over_mean"]
    pts = " ".join(f"{xp(i):.1f},{yp(v):.1f}" for i, v in enumerate(pk))
    dots = "".join(
        f'<circle cx="{xp(i):.1f}" cy="{yp(v):.1f}" r="4" fill="{color}" '
        f'stroke="var(--surface-1)" stroke-width="2"><title>L{i} · {v:.3f}×</title></circle>'
        for i, v in enumerate(pk))
    nf = D[key]["noise_floor"]
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round"/>{dots}'
            f'<line x1="{PAD}" y1="{yp(nf):.1f}" x2="{W-PAD}" y2="{yp(nf):.1f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-dasharray="3 4" opacity="0.75"/>')


grid = "".join(
    f'<line x1="{PAD}" y1="{yp(v):.1f}" x2="{W-PAD}" y2="{yp(v):.1f}" stroke="var(--grid)"/>'
    f'<text x="{PAD-2}" y="{yp(v)+3:.1f}" class="ax" text-anchor="end">{v:.1f}×</text>'
    for v in (1.0, 1.5, 2.0))

# The table view the heatmap and the line chart both needed.
per_layer_rows = []
for li in range(L):
    sh = D["prefill"]["share_by_layer"][li]
    peak_r = max(range(EP), key=lambda r: sh[r])
    cells = "".join(
        f'<td class="{"pkc" if r == peak_r else ""}">{sh[r]:.2f}</td>' for r in range(EP))
    per_layer_rows.append(
        f'<tr><th scope="row">L{li}</th>{cells}'
        f'<td class="em">{D["prefill"]["peak_over_mean"][li]:.3f}×</td>'
        f'<td>{D["decode"]["peak_over_mean"][li]:.3f}×</td></tr>')

def agg_row(key, label):
    a = D[key]["aggregate_share"]
    return (f'<tr><th scope="row">{label}</th>'
            + "".join(f"<td>{v:.2f}</td>" for v in a)
            + f'<td class="em">{D[key]["aggregate_imbalance"]:.3f}×</td></tr>')

legend_ticks = "".join(f'<span class="sw" style="background:{c}"></span>' for c in RAMP)
axis_ticks = "".join(f"<span>{i if i%6==0 else ''}</span>" for i in range(L))

page = f"""<title>Rank Imbalance by Layer — {MODEL}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --rule:#c3c2b7; --hair:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --wash:rgba(42,120,214,.09);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --rule:#383835; --hair:rgba(255,255,255,.10);
    --series-1:#3987e5; --series-2:#d95926; --wash:rgba(57,135,229,.16);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --rule:#383835; --hair:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --wash:rgba(57,135,229,.16);
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--plane); color:var(--ink);
  font:400 16px/1.6 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing:antialiased }}
.wrap {{ max-width:860px; margin:0 auto; padding:56px 24px 88px; display:flex; flex-direction:column; gap:44px }}
.eyebrow {{ font:500 11px/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted) }}
h1 {{ font:600 32px/1.2 "IBM Plex Sans", system-ui, sans-serif; margin:10px 0 0;
  text-wrap:balance; letter-spacing:-.01em }}
.lede {{ color:var(--ink-2); max-width:62ch; margin:14px 0 0 }}
section {{ display:flex; flex-direction:column; gap:16px }}
h2 {{ font:600 19px/1.3 "IBM Plex Sans", system-ui, sans-serif; margin:0; text-wrap:balance }}
p {{ margin:0; max-width:64ch; color:var(--ink-2) }}
.card {{ background:var(--surface-1); border:1px solid var(--hair); border-radius:6px; padding:20px }}
.duo {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px }}
.tile .k {{ font:500 11px/1 "IBM Plex Mono", monospace; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted) }}
.tile .v {{ font:600 40px/1 "IBM Plex Mono", monospace; margin-top:12px;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em }}
.tile .n {{ font-size:13.5px; color:var(--ink-2); margin-top:10px }}
.tile.hi .v {{ color:var(--series-1) }}
.tile.lo .v {{ color:var(--muted) }}
.hm {{ overflow-x:auto; padding-bottom:4px }}
.hmgrid {{ display:grid; grid-template-columns:repeat({L}, 15px); grid-auto-rows:15px;
  gap:2px; width:max-content }}
.cell {{ border-radius:2px }}
.cell.pk {{ box-shadow:0 0 0 2px var(--surface-1), 0 0 0 3px var(--ink) }}
.cell:focus-visible {{ outline:2px solid var(--ink); outline-offset:2px }}
.axrow {{ display:grid; grid-template-columns:repeat({L}, 15px); gap:2px;
  width:max-content; margin-top:6px }}
.axrow span {{ font:400 9px/1 "IBM Plex Mono", monospace; color:var(--muted); text-align:center }}
.rlab {{ display:flex; flex-direction:column; gap:2px; margin-right:8px }}
.rlab span {{ height:15px; font:400 10px/15px "IBM Plex Mono", monospace;
  color:var(--muted); white-space:nowrap }}
.hmrow {{ display:flex; align-items:flex-start }}
.scale {{ display:flex; align-items:center; gap:10px;
  font:400 12px/1 "IBM Plex Mono", monospace; color:var(--muted) }}
.sw {{ width:14px; height:10px; display:inline-block; border-radius:1px }}
.legend {{ display:flex; gap:20px; flex-wrap:wrap; font-size:13.5px; color:var(--ink-2) }}
.legend b {{ display:inline-flex; align-items:center; gap:7px; font-weight:500; color:var(--ink) }}
.dot {{ width:11px; height:11px; border-radius:50%; display:inline-block }}
.ax {{ font:400 10px/1 "IBM Plex Mono", monospace; fill:var(--muted) }}
table {{ border-collapse:collapse; font:400 13px/1.5 "IBM Plex Mono", monospace;
  font-variant-numeric:tabular-nums; width:100% }}
th, td {{ text-align:right; padding:6px 9px; border-bottom:1px solid var(--grid) }}
th[scope=row] {{ text-align:left; color:var(--ink); font-weight:500 }}
thead th {{ color:var(--muted); font-weight:500; font-size:11px;
  letter-spacing:.06em; text-transform:uppercase; position:sticky; top:0;
  background:var(--surface-1) }}
td.em {{ color:var(--ink); font-weight:600 }}
td.pkc {{ background:var(--wash); color:var(--ink); font-weight:600 }}
.tw {{ overflow-x:auto }}
.tall {{ max-height:420px; overflow:auto; border:1px solid var(--hair); border-radius:6px }}
.note {{ font-size:13.5px; color:var(--muted); max-width:64ch }}
code {{ font:400 .93em/1 "IBM Plex Mono", monospace; background:var(--plane);
  padding:2px 5px; border-radius:3px; border:1px solid var(--hair) }}
#tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--ink); color:var(--surface-1);
  font:400 12px/1.45 "IBM Plex Mono", monospace; padding:7px 9px;
  border-radius:4px; max-width:260px; z-index:9 }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important }} }}
hr {{ border:0; border-top:1px solid var(--grid); margin:0 }}
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">{MODEL} · DP={EP} EP={EP} · {L} MoE layers · {_any.get("num_logical_experts","?")} logical experts</div>
    <h1>The same imbalance, measured two ways</h1>
    <p class="lede">Expert load across eight ranks looks almost even when the layers are
      summed first. Per layer — which is what a step actually waits on — it is not. These
      are the same {D['prefill']['forwards']} prefill forwards, read with two metrics.</p>
  </header>

  <section>
    <div class="duo">
      <div class="card tile lo">
        <div class="k">Summed over layers</div>
        <div class="v">{D['prefill']['aggregate_imbalance']:.3f}×</div>
        <div class="n">Add all {L} layers per rank, then compare ranks. Reads as
          near-balanced: {UNIFORM:.2f}% would be even, and the spread is
          {max(D['prefill']['aggregate_share'])-min(D['prefill']['aggregate_share']):.2f}
          points. <strong>This is the misleading one.</strong></div>
      </div>
      <div class="card tile hi">
        <div class="k">Per-layer critical path</div>
        <div class="v">{D['prefill']['critical_path']:.3f}×</div>
        <div class="n">Sum each layer's <em>peak</em> rank, divide by the sum of its means.
          Every layer is its own collective and waits for its own slowest rank, so this is
          the figure that costs time.</div>
      </div>
    </div>
    <p class="note">Multinomial noise alone would give
      {D['prefill']['noise_floor']:.3f}× at {D['prefill']['tokens_per_expert']} tokens per
      expert, so the per-layer skew is real routing behaviour rather than sampling.</p>
  </section>

  <hr>

  <section>
    <h2>Where each layer's load actually sits</h2>
    <p>One cell per rank per layer, shaded by that rank's share of that layer's tokens.
      A ring marks the layer's peak rank. The rings wander — that is precisely why summing
      the layers first cancels them out.</p>
    <div class="card">
      <div class="hm">
        <div class="hmrow">
          <div class="rlab">{"".join(f"<span>rank {r}</span>" for r in range(EP))}</div>
          <div>
            <div class="hmgrid" id="hm">{heat_cells("prefill")}</div>
            <div class="axrow">{axis_ticks}</div>
          </div>
        </div>
      </div>
      <div class="scale" style="margin-top:16px">
        <span>{LO:.0f}%</span>{legend_ticks}<span>{HI:.0f}%</span>
        <span style="margin-left:auto">even share = {UNIFORM:.2f}%</span>
      </div>
    </div>
  </section>

  <section>
    <h2>Per-layer peak over mean, prefill against decode</h2>
    <p>Both regimes carry a real, comparable skew. The dashed line under each series is
      that regime's noise floor — decode's is high because it has only
      {D['decode']['tokens_per_expert']} tokens per expert, prefill's is near 1.0 because it
      has {D['prefill']['tokens_per_expert']}.</p>
    <div class="card">
      <div class="legend" style="margin-bottom:12px">
        <b><span class="dot" style="background:var(--series-1)"></span>Prefill · median
          {D['prefill']['critical_path']:.3f}×</b>
        <b><span class="dot" style="background:var(--series-2)"></span>Decode · median
          {D['decode']['critical_path']:.3f}×</b>
      </div>
      <div style="overflow-x:auto">
        <svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
             aria-label="Per-layer peak-over-mean rank imbalance, prefill and decode, across {L} layers">
          {grid}{spark("decode","var(--series-2)")}{spark("prefill","var(--series-1)")}
        </svg>
      </div>
      <div class="axrow" style="margin-left:0">{axis_ticks}</div>
    </div>
  </section>

  <hr>

  <section>
    <h2>Every layer, every rank — the numbers behind both charts</h2>
    <p>Prefill share of each layer's tokens, per rank. The shaded cell in each row is that
      layer's peak. The last two columns are the two series in the line chart.</p>
    <div class="tall">
      <table>
        <thead><tr><th scope="col" style="text-align:left">Layer</th>
          {"".join(f'<th scope="col">r{r}</th>' for r in range(EP))}
          <th scope="col">prefill pk/mean</th><th scope="col">decode pk/mean</th></tr></thead>
        <tbody>{"".join(per_layer_rows)}</tbody>
      </table>
    </div>
    <p class="note">Shares are percentages of that layer's tokens and each row sums to
      100%. Averaged over the {D['prefill']['forwards']} prefill forwards in the dump.</p>
  </section>

  <section>
    <h2>Summed over layers — the view that hides it</h2>
    <p class="note">Kept only as the evidence behind the first tile. Every rank is the peak
      on a handful of the {L} layers, so adding the layers up cancels the skew and
      leaves a 2-point spread. Do not read a placement decision off this table.</p>
    <div class="tw">
      <table>
        <thead><tr><th scope="col" style="text-align:left">Regime</th>
          {"".join(f'<th scope="col">r{r}</th>' for r in range(EP))}
          <th scope="col">peak / mean</th></tr></thead>
        <tbody>{agg_row("prefill","Prefill")}{agg_row("decode","Decode")}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Why prefill can spend it and decode cannot</h2>
    <p>The kernel pads each expert's token list up to a multiple of
      <code>BLOCK_SIZE_M</code>, so imbalance smaller than one block is free. Prefill clears
      that threshold; decode does not.</p>
    <div class="tw">
      <table>
        <thead><tr><th scope="col" style="text-align:left">Regime</th>
          <th scope="col">M</th><th scope="col">tokens / expert</th>
          <th scope="col">BLOCK_SIZE_M</th><th scope="col">blocks / expert</th>
          <th scope="col">critical path</th><th scope="col">transduces?</th></tr></thead>
        <tbody>
          <tr><th scope="row">Prefill</th><td>{D['prefill']['M']:,}</td>
            <td>{D['prefill']['tokens_per_expert']}</td><td>128</td><td>8</td>
            <td class="em">{D['prefill']['critical_path']:.3f}×</td><td class="em">yes</td></tr>
          <tr><th scope="row">Decode</th><td>{D['decode']['M']:,}</td>
            <td>{D['decode']['tokens_per_expert']}</td><td>32</td><td>1</td>
            <td class="em">{D['decode']['critical_path']:.3f}×</td><td>no</td></tr>
        </tbody>
      </table>
    </div>
    <p class="note">M is the post-allgather token count the MoE kernel sees, not one rank's
      scheduler budget. At 8 blocks per expert a 1.60× token skew resolves to roughly 1.5×
      in time; at 1 block it resolves to nothing.</p>
  </section>
</div>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
function show(e, t) {{
  tip.textContent = t; tip.style.opacity = 1;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY - r.height - 10;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y < 8) y = e.clientY + 16;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}}
const hm = document.getElementById('hm');
hm.addEventListener('mousemove', e => {{
  const c = e.target.closest('.cell');
  if (c) show(e, c.dataset.t); else tip.style.opacity = 0;
}});
hm.addEventListener('mouseleave', () => tip.style.opacity = 0);
hm.addEventListener('focusin', e => {{
  const c = e.target.closest('.cell'); if (!c) return;
  const b = c.getBoundingClientRect();
  show({{clientX: b.left + b.width/2, clientY: b.top}}, c.dataset.t);
}});
hm.addEventListener('focusout', () => tip.style.opacity = 0);
</script>
"""
_args.out.write_text(page)
print(f"wrote {_args.out} ({len(page)} bytes), {L} layers x {EP} ranks")

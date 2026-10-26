"""Risk assessment page (the 5×5 register and matrix) and the safety assistant drawer.

The register is deterministic (`pipeline/output/risk.py`). The assistant, a slide-over
panel reachable from every section, talks to the
hosted API when the page is served by it (`<meta name="safety-api">`, injected by
`app/cloud.py`); opened as a file, it still answers the suggested questions from the
register, prepared when the report was written.
"""

from __future__ import annotations

import json
from html import escape

from pipeline.output.facts import SUGGESTED, Fact, incident_ids, register_answer
from pipeline.output.risk import LIKELIHOOD, RiskAssessment, RiskRow, band
from pipeline.output.run_bundle import RunBundle

BAND_SLOT = {"very high": "critical", "high": "serious", "medium": "warning", "low": "good"}
SEVERITY_WORD = {5: "critical", 4: "high", 3: "medium", 2: "low", 1: "minimal"}


def page(bundle: RunBundle, facts: list[Fact], *, head: str, card) -> str:  # noqa: ANN001
    risk = bundle.risk
    short = incident_ids(bundle)
    register = "".join(_row(row, short) for row in risk.rows) or (
        '<p class="empty">No hazard condition held in this clip.</p>'
    )
    top = risk.rows[0] if risk.rows else None
    kpis = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>'
        for label, value, note in (
            ("Hazards rated", str(len(risk.rows)), "held at least briefly in this clip"),
            (
                "Highest risk",
                f"{top.rating}/25" if top else "—",
                escape(f"{top.rule_id.value} {top.band}") if top else "nothing rated",
            ),
            ("Not assessed", str(len(risk.not_assessed)), "rules this clip could not check"),
            ("Workers seen", str(bundle.workers_seen), "in view for at least 3 s"),
        )
    )
    return f"""{head}
<div class="kpis">{kpis}</div>
<div class="stack-gap">
  {card("Risk register", register, "likelihood × severity, highest first")}
  <div class="grid cols-2">
    {card("Risk matrix", _matrix(risk), "5 × 5")}
    {card("Not rated", _unrated(risk))}
  </div>
  <details class="card fold"><summary>How the ratings are made</summary>
    <div class="card-body">{_method(risk)}</div></details>
</div>"""


def drawer(bundle: RunBundle, facts: list[Fact]) -> str:
    """The assistant as a slide-over panel, opened from any section of the report."""
    short = incident_ids(bundle)
    prepared = {}
    for question in SUGGESTED:
        text, cited = register_answer(bundle, facts, question)
        prepared[question] = {"text": text, "citations": [f.id for f in cited]}
    data = {
        "prepared": prepared,
        "incidents": {sid: iid for iid, sid in short.items()},
        "facts": {f.id: f.text for f in facts},
    }
    payload = json.dumps(data, ensure_ascii=False).replace("<", chr(92) + "u003c")  # no </script>
    chips = "".join(
        f'<button class="btn sm" type="button" data-ask="{escape(q)}">{escape(q)}</button>'
        for q in SUGGESTED
    )
    return f"""<div class="scrim" aria-hidden="true"></div>
<aside class="chat" id="chat" aria-label="Safety assistant" aria-hidden="true">
  <div class="chat-head">
    <div class="who"><span class="ai-dot" aria-hidden="true"></span><div><b>Safety assistant</b>
      <div class="hint" id="chat-mode">offline</div></div></div>
    <button class="btn icon ghost" type="button" data-ask-close aria-label="Close the assistant">✕</button>
  </div>
  <div class="msgs" id="chat-msgs" aria-live="polite">
    <div class="msg bot"><div class="bubble">Ask about this clip's hazards, workers, incidents or what
    to fix first. Every claim cites a fact from the run; click a citation to see it.</div></div>
  </div>
  <div class="suggest" id="chat-suggest">{chips}</div>
  <form id="chat-form" autocomplete="off">
    <input id="chat-q" type="text" maxlength="800" placeholder="Ask about risks, workers, incidents…"
      aria-label="Your question">
    <button class="btn primary" type="submit">Send</button>
  </form>
</aside>
<script type="application/json" id="chat-data">{payload}</script>"""


def _row(row: RiskRow, short: dict[str, str]) -> str:
    slot = BAND_SLOT[row.band]
    links = " ".join(
        f'<button class="link" type="button" data-open="{escape(iid)}">{escape(short[iid])}</button>'
        for iid in row.incident_ids[:8]
        if iid in short
    )
    more = f" +{len(row.incident_ids) - 8}" if len(row.incident_ids) > 8 else ""
    meta = []
    if row.top_workers:
        noun = "worker" if len(row.top_workers) == 1 else "workers"
        meta.append(f"Most exposed: {noun} {escape(', '.join(row.top_workers))}")
    if row.areas:
        meta.append(f"Where: {escape(', '.join(row.areas))}")
    meta.append(f"{row.exposure_s:.1f} worker-seconds of exposure")
    controls = "".join(
        f'<li><span class="lvl">{escape(level)}</span>{escape(text)}</li>'
        for level, text in row.controls
    )
    return f"""<article class="risk-row" style="--sev:var(--{slot})">
  <div class="score"><b>{row.rating}</b><span>/25</span>
    <span class="chip"><span class="sw" style="background:var(--{slot})"></span>{escape(row.band.capitalize())}</span></div>
  <div class="what">
    <div class="t">{escape(row.rule_id.value)} · {escape(row.title)}</div>
    <p class="hz">{escape(row.hazard)}.</p>
    <div class="lxs">Likelihood <b>{row.likelihood_score}</b> {escape(row.likelihood)} ×
      severity <b>{row.severity_score}</b> {escape(row.severity)}</div>
    <p class="why">{escape(row.why)}</p>
    <div class="m">{" · ".join(meta)}</div>
    {f'<div class="m">Incidents: {links}{more}</div>' if links else ""}
    <details><summary>Controls, in hierarchy-of-controls order</summary><ol class="controls">{controls}</ol></details>
  </div>
</article>"""


def _matrix(risk: RiskAssessment) -> str:
    placed: dict[tuple[int, int], list[str]] = {}
    for row in risk.rows:
        placed.setdefault((row.likelihood_score, row.severity_score), []).append(row.rule_id.value)
    cells = []
    for likelihood in range(5, 0, -1):
        cells.append(
            f'<div class="axis l">{likelihood}<span>{escape(LIKELIHOOD[likelihood])}</span></div>'
        )
        for severity in range(1, 6):
            slot = BAND_SLOT[band(likelihood * severity)]
            rules = "".join(
                f'<span class="dot-rule">{escape(r)}</span>'
                for r in placed.get((likelihood, severity), [])
            )
            cells.append(
                f'<div class="cell" style="--tint:var(--{slot})" '
                f'title="Likelihood {likelihood} × severity {severity} = {likelihood * severity}">'
                f"{rules}</div>"
            )
    cells.append('<div class="axis"></div>')
    cells += [f'<div class="axis s">{s}<span>{SEVERITY_WORD[s]}</span></div>' for s in range(1, 6)]
    legend = "".join(
        f'<span><span class="sw" style="background:var(--{slot})"></span>{word}</span>'
        for word, slot in (
            ("Low ≤4", "good"),
            ("Medium 5–10", "warning"),
            ("High 12–16", "serious"),
            ("Very high ≥20", "critical"),
        )
    )
    return (
        '<div class="matrix" role="img" aria-label="Risk matrix: likelihood by severity">'
        + "".join(cells)
        + '</div><div class="matrix-cap"><span>↑ likelihood</span><span>severity →</span></div>'
        + f'<div class="legend">{legend}</div>'
    )


def _unrated(risk: RiskAssessment) -> str:
    assessed = "".join(
        f"<li><b>{escape(rule.value)}</b> — {escape(why)} "
        f'<span class="muted">({escape(reason.replace("_", " "))})</span></li>'
        for rule, reason, why in risk.not_assessed
    )
    clear = "".join(
        f"<li><b>{escape(rule.value)}</b> — checked for the whole clip, not observed</li>"
        for rule, _ in risk.not_observed
    )
    return (
        '<p class="section-title" style="margin-top:0">Not assessed — not the same as no risk</p>'
        f'<ul class="clean">{assessed or "<li>Every rule could be assessed.</li>"}</ul>'
        '<p class="section-title">Checked, not observed</p>'
        f'<ul class="clean">{clear or "<li>None.</li>"}</ul>'
    )


def _method(risk: RiskAssessment) -> str:
    notes = "".join(f"<li>{escape(n)}</li>" for n in risk.notes)
    return f"""<ul class="clean">
<li><b>Severity</b> is the rule catalogue's: critical 5, high 4, medium 3, low 2.</li>
<li><b>Likelihood</b> is judged from this clip: 2 when the condition only held briefly, 3 for one
confirmed incident, 4 for 2–4 incidents or at least 10 % of workers exposed, 5 for 5 or more
incidents or at least 25 % exposed. All-weak evidence lowers it one step.</li>
<li><b>Controls</b> follow the hierarchy of controls (eliminate, engineering, administrative,
PPE). They are general good practice, not legal requirements.</li>
{notes}</ul>"""


STYLE = """

.risk-row { display:grid; grid-template-columns:96px minmax(0,1fr); gap:16px; padding:14px 2px;
  border-bottom:1px solid var(--line); }
.risk-row:last-child { border-bottom:none; }
.risk-row .score { display:flex; flex-direction:column; align-items:flex-start; gap:4px;
  border-left:4px solid var(--sev); padding-left:12px; }
.risk-row .score b { font-family:var(--display); font-size:1.9rem; line-height:1; font-variant-numeric:tabular-nums; }
.risk-row .score > span:not(.chip) { color:var(--text-3); font-size:.8rem; margin-top:-2px; }
.risk-row .t { font-weight:650; }
.risk-row .hz { margin:2px 0 6px; color:var(--text-2); font-size:.86rem; }
.risk-row .lxs { font-size:.82rem; color:var(--text-2); }
.risk-row .why { margin:4px 0; font-size:.86rem; }
.risk-row .m { color:var(--text-3); font-size:.8rem; margin-top:2px; }
.risk-row details summary { margin-top:6px; }
ol.controls { margin:8px 0 0; padding-left:20px; font-size:.86rem; }
ol.controls li + li { margin-top:4px; }
ol.controls .lvl { display:inline-block; min-width:104px; color:var(--text-3); font-size:.74rem;
  text-transform:uppercase; letter-spacing:.05em; }
button.link { border:none; background:none; color:var(--accent); cursor:pointer; padding:0 2px; font-size:inherit; }
button.link:hover { text-decoration:underline; }
.matrix { display:grid; grid-template-columns:96px repeat(5, minmax(0,1fr)); gap:3px; }
.matrix .cell { aspect-ratio:1.5; border-radius:6px; display:flex; flex-wrap:wrap; gap:3px; align-content:center;
  justify-content:center; background:color-mix(in srgb, var(--tint) 30%, var(--surface)); opacity:.85; }
.matrix .axis { font-size:.74rem; color:var(--text-2); display:flex; flex-direction:column; justify-content:center; }
.matrix .axis span { color:var(--text-3); font-size:.68rem; }
.matrix .axis.s { align-items:center; text-align:center; }
.dot-rule { background:var(--text); color:var(--surface); font-size:.68rem; font-weight:700; border-radius:999px;
  padding:1px 6px; }
.matrix-cap { display:flex; justify-content:space-between; color:var(--text-3); font-size:.72rem; margin-top:6px; }
.chat { position:fixed; top:0; right:0; bottom:0; width:min(440px, 100vw); z-index:45; display:flex; flex-direction:column;
  background:var(--surface); border-left:1px solid var(--line-2); box-shadow:var(--shadow-lg);
  transform:translateX(104%); transition:transform .25s cubic-bezier(.2,.8,.2,1); visibility:hidden; }
body.chat-open .chat { transform:none; visibility:visible; }
.chat .chat-head { display:flex; align-items:center; gap:10px; padding:14px 14px 14px 18px; border-bottom:1px solid var(--line); }
.chat .who { display:flex; align-items:center; gap:10px; margin-right:auto; }
.chat .who b { font-family:var(--display); font-size:1rem; }
.chat .hint { color:var(--text-3); font-size:.74rem; }
.chat .ai-dot { width:30px; height:30px; border-radius:9px; flex:none; background:conic-gradient(from 210deg, var(--accent), #8b5cf6, var(--accent));
  box-shadow:0 0 0 3px var(--accent-soft); }
.chat .msgs { flex:1; overflow:auto; padding:14px 16px; display:flex; flex-direction:column; gap:10px; min-height:0; }
.chat .msg { display:flex; flex-direction:column; max-width:92%; }
.chat .msg.user { align-self:flex-end; align-items:flex-end; }
.chat .bubble { padding:10px 13px; border-radius:14px; line-height:1.5; background:var(--surface-2); font-size:.88rem; }
.chat .msg.user .bubble { background:var(--accent); color:#fff; border-bottom-right-radius:4px; }
.chat .msg.bot .bubble { border-bottom-left-radius:4px; }
.chat .bubble ul { margin:4px 0; padding-left:18px; }
.chat .bubble li + li { margin-top:4px; }
.chat .meta { color:var(--text-3); font-size:.72rem; margin-top:3px; }
.chat .cite { border:1px solid var(--line); background:var(--surface); color:var(--accent); border-radius:6px;
  font-size:.72rem; padding:0 5px; cursor:pointer; margin:0 1px; }
.chat .fact { margin-top:6px; padding:8px 10px; border-left:3px solid var(--accent); background:var(--surface);
  border-radius:6px; font-size:.8rem; color:var(--text-2); }
.chat .typing { color:var(--text-3); font-size:.82rem; }
.chat .suggest { display:flex; flex-wrap:wrap; gap:6px; padding:0 16px 10px; }
.chat form { display:flex; gap:8px; padding:12px 16px 16px; border-top:1px solid var(--line); }
.chat input { flex:1; min-width:0; border:1px solid var(--line-2); border-radius:10px; padding:9px 12px;
  background:var(--surface); color:var(--text); font:inherit; }

@media (max-width:560px) { .chat { width:100vw; } }
@media (max-width:560px) { .risk-row { grid-template-columns:1fr; } .matrix { grid-template-columns:64px repeat(5, minmax(0,1fr)); } }
"""

SCRIPT = r"""
(() => {
const box = document.getElementById('chat');
if (!box) return;
const data = JSON.parse(document.getElementById('chat-data').textContent);
const apiMeta = document.querySelector('meta[name="safety-api"]');
const api = apiMeta ? apiMeta.content : null;
const msgs = document.getElementById('chat-msgs');
const input = document.getElementById('chat-q');
const mode = document.getElementById('chat-mode');
const history = [];
mode.textContent = api ? 'Live · every claim checked against this run' : 'Offline · suggested questions only';
if (!api) input.placeholder = 'Offline: pick a suggested question above';
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function inline(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?(?:\s*,\s*[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?)*)\]/g,
      (_, ids) => ids.split(/\s*,\s*/).map(id => data.facts[id]
        ? `<button type="button" class="cite" data-cite="${id}">${id}</button>` : esc('[' + id + ']')).join(''));
}
function render(text) {
  const out = []; let list = false;
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    const bullet = /^([-*•]|\d+[.)])\s+/.test(line);
    if (bullet && !list) { out.push('<ul>'); list = true; }
    if (!bullet && list) { out.push('</ul>'); list = false; }
    if (!line) continue;
    out.push(bullet ? '<li>' + inline(line.replace(/^([-*•]|\d+[.)])\s+/, '')) + '</li>' : '<p style="margin:2px 0">' + inline(line) + '</p>');
  }
  if (list) out.push('</ul>');
  return out.join('');
}
function fill(m, html, meta) {
  m.innerHTML = '<div class="bubble">' + html + '</div>' + (meta ? '<div class="meta">' + esc(meta) + '</div>' : '');
  msgs.scrollTop = msgs.scrollHeight; return m;
}
function add(role, html, meta) {
  const m = document.createElement('div'); m.className = 'msg ' + role;
  msgs.appendChild(m); return fill(m, html, meta);
}
let busy = false;
const send = box.querySelector('form button');
function describe(reply) {
  if (reply.source === 'model') return `Verified against the run · ${reply.model || 'model'} · ${reply.latency_s}s`;
  if (reply.source === 'scope') return 'Answered by the assistant rules, no model';
  return (reply.failures || []).length ? 'Answered from the run data (the model draft failed a check)' : 'Answered from the run data';
}
async function ask(q) {
  q = q.trim(); if (!q || busy) return;
  add('user', esc(q));
  if (!api) {
    const p = data.prepared[q];
    if (p) add('bot', render(p.text), 'Answered from the run data (offline)');
    else add('bot', 'Free-text questions need the hosted app. Pick a suggested question, or open this report from the hosted demo.', '');
    return;
  }
  const wait = add('bot', '<span class="typing">Checking the run…</span>');
  busy = true; send.disabled = true;
  try {
    const r = await fetch(api + '/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, history: history.slice(-6) }) });
    if (!r.ok) throw new Error(r.status === 429 ? 'Too many questions; wait a minute.' : 'The assistant is unavailable (' + r.status + ').');
    const reply = await r.json();
    fill(wait, render(reply.text), describe(reply));
    history.push({ role: 'user', content: q }, { role: 'assistant', content: reply.text });
  } catch (e) {
    const p = data.prepared[q];
    fill(wait, p ? render(p.text) : esc(e.message || 'The assistant is unavailable.'), p ? 'Answered from the run data (offline)' : '');
  } finally {
    busy = false; send.disabled = false;
  }
}
document.getElementById('chat-form').addEventListener('submit', e => { e.preventDefault(); const q = input.value; input.value = ''; ask(q); });
box.querySelectorAll('[data-ask]').forEach(b => b.addEventListener('click', () => ask(b.dataset.ask)));
msgs.addEventListener('click', e => {
  const b = e.target.closest('[data-cite]'); if (!b) return;
  const id = b.dataset.cite;
  if (data.incidents[id] && window.openIncident) { if (window.closeChat) window.closeChat(); window.openIncident(data.incidents[id]); return; }
  const bubble = b.closest('.bubble'); const open = bubble.querySelector('.fact[data-for="' + id + '"]');
  if (open) { open.remove(); return; }
  const f = document.createElement('div'); f.className = 'fact'; f.dataset.for = id;
  f.innerHTML = '<b>' + esc(id) + '</b> ' + esc(data.facts[id] || ''); bubble.appendChild(f);
});
})();
"""

"""Stylesheet and script for the offline report (one file, no network, no framework).

Tokens follow the status palette (good / warning / serious / critical) and a neutral
surface scale, defined for light and dark. Layout: a sticky top bar, a sidebar of
sections (tabs on narrow screens), one section shown at a time, and an inbox-style
incident view (list + detail).
"""

STYLE = """
:root { color-scheme: light;
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#f1f3f6; --surface-3:#e7eaef; --line:#e2e5ea; --line-2:#d3d8e0;
  --text:#0e1116; --text-2:#465061; --text-3:#7a8394; --accent:#2563eb; --accent-soft:#e8efff; --accent-ink:#1d4fd8;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --neutral:#8a8f99; --muted:#c5cad3; --series:#2563eb; --hazard:#f5b301;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 8px rgba(16,24,40,.04);
  --shadow-lg:0 12px 40px rgba(16,24,40,.16);
  --font:"Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --display:"Space Grotesk", "Inter", "Segoe UI", system-ui, sans-serif;
  --mono:"JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  --radius:14px; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  color-scheme: dark; --bg:#0c0e12; --surface:#13161b; --surface-2:#1a1e25; --surface-3:#222731; --line:#242a33;
  --line-2:#2f3642; --text:#eef1f5; --text-2:#b3bbc7; --text-3:#7d8796; --accent:#5b9dff; --accent-soft:rgba(91,157,255,.13);
  --accent-ink:#8dbaff; --neutral:#8a919c; --muted:#3c434e; --series:#5b9dff;
  --shadow:0 1px 0 rgba(255,255,255,.02) inset; --shadow-lg:0 16px 48px rgba(0,0,0,.55); } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0c0e12; --surface:#13161b; --surface-2:#1a1e25; --surface-3:#222731;
  --line:#242a33; --line-2:#2f3642; --text:#eef1f5; --text-2:#b3bbc7; --text-3:#7d8796; --accent:#5b9dff;
  --accent-soft:rgba(91,157,255,.13); --accent-ink:#8dbaff; --neutral:#8a919c; --muted:#3c434e; --series:#5b9dff;
  --shadow:0 1px 0 rgba(255,255,255,.02) inset; --shadow-lg:0 16px 48px rgba(0,0,0,.55); }
* { box-sizing:border-box; }
[hidden] { display:none !important; }
html, body { margin:0; }
body { background:var(--bg); color:var(--text); font:14px/1.55 var(--font); -webkit-font-smoothing:antialiased;
  font-feature-settings:"cv11", "ss01"; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
button { font:inherit; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:6px; }
code, .mono { font-family:var(--mono); font-size:.84em; }
code { background:var(--surface-2); padding:1px 6px; border-radius:6px; border:1px solid var(--line); }
h1, h2, h3 { font-family:var(--display); letter-spacing:-.01em; }

/* ---- app shell */
.topbar { position:sticky; top:0; z-index:30; display:flex; align-items:center; gap:14px; height:60px; padding:0 20px;
  background:color-mix(in srgb, var(--surface) 86%, transparent); backdrop-filter:saturate(1.4) blur(12px);
  border-bottom:1px solid var(--line); }
.brand { display:flex; align-items:center; gap:9px; flex:none; }
.logo-mark { display:block; flex:none; }
.wordmark { font-family:var(--display); font-weight:700; font-size:.98rem; letter-spacing:.05em; white-space:nowrap; }
.topbar .ident { min-width:0; display:flex; align-items:center; gap:12px; }
.topbar .divider { width:1px; height:22px; background:var(--line-2); }
.topbar .title { font-weight:600; font-size:.92rem; line-height:1.2; white-space:nowrap; }
.topbar .meta { color:var(--text-3); font-size:.76rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-family:var(--mono); }
.topbar .spacer { flex:1; }
.topbar .back { color:var(--text-2); font-size:.84rem; display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
.btn { display:inline-flex; align-items:center; justify-content:center; gap:7px; background:var(--surface); color:var(--text);
  border:1px solid var(--line-2); border-radius:9px; padding:7px 13px; cursor:pointer; text-decoration:none;
  font-size:.84rem; font-weight:500; line-height:1.2; transition:background .15s, border-color .15s, color .15s; white-space:nowrap; }
.btn:hover { background:var(--surface-2); text-decoration:none; }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.btn.primary:hover { background:var(--accent-ink); border-color:var(--accent-ink); }
.btn.ghost { background:transparent; border-color:transparent; color:var(--text-2); }
.btn.ghost:hover { background:var(--surface-2); color:var(--text); }
.btn.icon { padding:7px; width:34px; height:34px; }
.btn svg { flex:none; }
details.menu { position:relative; }
details.menu > summary { list-style:none; }
details.menu > summary::-webkit-details-marker { display:none; }
details.menu .menu-pop { position:absolute; right:0; top:calc(100% + 6px); min-width:220px; background:var(--surface);
  border:1px solid var(--line-2); border-radius:12px; box-shadow:var(--shadow-lg); padding:6px; z-index:40; }
details.menu .menu-pop a { display:block; padding:8px 10px; border-radius:8px; color:var(--text); font-size:.85rem; }
details.menu .menu-pop a:hover { background:var(--surface-2); text-decoration:none; }
details.menu .menu-pop a span { display:block; color:var(--text-3); font-size:.75rem; }
.shell { display:grid; grid-template-columns:228px minmax(0,1fr); min-height:calc(100vh - 60px); }
.sidebar { position:sticky; top:60px; height:calc(100vh - 60px); padding:18px 12px; border-right:1px solid var(--line);
  background:var(--surface); overflow:auto; display:flex; flex-direction:column; }
.sidebar .group { color:var(--text-3); font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin:4px 10px 8px; }
.nav a { position:relative; display:flex; align-items:center; gap:10px; padding:8px 10px; border-radius:9px; color:var(--text-2);
  text-decoration:none; font-weight:500; margin-bottom:2px; transition:background .15s, color .15s; }
.nav a:hover { background:var(--surface-2); color:var(--text); }
.nav a[aria-current="page"] { background:var(--accent-soft); color:var(--accent); }
.nav a[aria-current="page"]::before { content:""; position:absolute; left:-12px; top:8px; bottom:8px; width:3px;
  border-radius:0 3px 3px 0; background:var(--accent); }
.nav .ico { width:18px; height:18px; display:grid; place-items:center; opacity:.85; }
.nav .count { margin-left:auto; font-size:.72rem; color:var(--text-2); background:var(--surface-2); border:1px solid var(--line);
  border-radius:999px; padding:0 7px; font-family:var(--mono); }
.sidebar .ask-side { margin:14px 4px 0; }
.sidebar .ask-side .btn { width:100%; }
main { padding:30px 36px 40px; max-width:1360px; width:100%; }
section.page { display:none; }
section.page.active { display:block; animation:enter .22s ease-out; }
@keyframes enter { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) { section.page.active { animation:none; } }
.page-head { display:flex; align-items:flex-end; gap:16px; margin-bottom:22px; flex-wrap:wrap; }
.page-head h1 { font-size:1.6rem; margin:0; font-weight:700; }
.page-head p { margin:6px 0 0; color:var(--text-2); max-width:680px; }
.page-head .actions { margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.page-foot { margin-top:40px; padding-top:16px; border-top:1px solid var(--line); color:var(--text-3); font-size:.78rem;
  display:flex; gap:8px 18px; flex-wrap:wrap; }

/* ---- verdict (overview) */
.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:18px; padding:26px 28px;
  margin-bottom:16px; background:
    radial-gradient(600px 220px at 100% 0%, color-mix(in srgb, var(--sev) 18%, transparent), transparent 70%),
    linear-gradient(var(--surface), var(--surface)); box-shadow:var(--shadow); }
.hero::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--sev); }
.hero .eyebrow { color:var(--text-3); font-size:.78rem; font-family:var(--mono); letter-spacing:.02em; }
.hero h1 { margin:8px 0 6px; font-size:1.75rem; line-height:1.2; font-weight:700; max-width:900px; }
.hero h1 em { font-style:normal; color:var(--sev); }
.hero p { margin:0; color:var(--text-2); max-width:760px; }
.hero .cta { display:flex; gap:8px; flex-wrap:wrap; margin-top:18px; }

/* ---- building blocks */
.card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); }
.card > .card-head { display:flex; align-items:center; gap:10px; padding:14px 18px; border-bottom:1px solid var(--line); }
.card > .card-head h2 { font-size:.98rem; margin:0; font-weight:600; }
.card > .card-head .hint { color:var(--text-3); font-size:.78rem; margin-left:auto; }
.card > .card-body { padding:16px 18px; }
details.card.fold > summary { list-style:none; cursor:pointer; display:flex; align-items:center; gap:10px; padding:14px 18px;
  font-family:var(--display); font-weight:600; font-size:.98rem; margin:0; color:var(--text); }
details.card.fold > summary::-webkit-details-marker { display:none; }
details.card.fold > summary::after { content:""; margin-left:auto; width:8px; height:8px; border-right:2px solid var(--text-3);
  border-bottom:2px solid var(--text-3); transform:rotate(45deg); transition:transform .15s; }
details.card.fold[open] > summary { border-bottom:1px solid var(--line); }
details.card.fold[open] > summary::after { transform:rotate(-135deg); }
details.card.fold > .card-body { padding:16px 18px; }
details.card.fold > summary .hint { color:var(--text-3); font-size:.78rem; font-family:var(--font); font-weight:400; }
.grid { display:grid; gap:16px; align-items:start; }
.grid.cols-2 { grid-template-columns:repeat(2, minmax(0,1fr)); }
.grid.cols-3-2 { grid-template-columns:minmax(0,3fr) minmax(0,2fr); }
.stack-gap > * + * { margin-top:16px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:16px; }
.kpi { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); }
.kpi .label { color:var(--text-3); font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.kpi .value { font-family:var(--display); font-size:1.8rem; font-weight:700; line-height:1.2; margin-top:4px;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.kpi .note { color:var(--text-3); font-size:.78rem; }
.chip { display:inline-flex; align-items:center; gap:6px; padding:2px 9px; border-radius:999px; font-size:.74rem;
  font-weight:600; background:var(--surface-2); color:var(--text); border:1px solid var(--line); white-space:nowrap; }
.chip i { font-style:normal; }
.sw { display:inline-block; width:8px; height:8px; border-radius:50%; flex:none; }
.muted { color:var(--text-3); }
.sub { color:var(--text-2); margin:0; }
.small { font-size:.8rem; }
.banner { display:flex; gap:12px; align-items:flex-start; background:var(--surface); border:1px solid var(--line);
  border-left:4px solid var(--warning); border-radius:10px; padding:10px 14px; color:var(--text-2); margin-bottom:16px; font-size:.86rem; }
table { width:100%; border-collapse:collapse; font-size:.85rem; }
th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--text-3); font-weight:600; font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; }
tbody tr:hover td { background:color-mix(in srgb, var(--surface-2) 60%, transparent); }
tr:last-child td { border-bottom:none; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
details > summary { cursor:pointer; color:var(--text-2); font-size:.84rem; margin-top:10px; }
.empty { color:var(--text-3); padding:20px 0; text-align:center; }

/* ---- replay */
video#video { width:100%; border-radius:10px; background:#000; display:block; aspect-ratio:16/9; }
svg text { fill:var(--text-3); font-size:11px; font-family:var(--mono); }
.lane-label { fill:var(--text-2); font-size:12px; font-weight:600; }
.span { cursor:pointer; transition:opacity .12s; }
.span:hover, .span:focus { opacity:.75; outline:none; }
.legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--text-2); font-size:.79rem; align-items:center; margin-top:8px; }
.legend > span { display:inline-flex; align-items:center; gap:6px; }
.hatch { background-image:repeating-linear-gradient(45deg, var(--muted) 0 2px, transparent 2px 6px); }

/* ---- priority list (overview) */
.prio { list-style:none; margin:0; padding:0; }
.prio li { display:grid; grid-template-columns:3px 1fr auto; gap:12px; align-items:center; padding:9px 6px;
  border-bottom:1px solid var(--line); cursor:pointer; border-radius:8px; transition:background .12s; }
.prio li:last-child { border-bottom:none; }
.prio li:hover { background:var(--surface-2); }
.prio .bar { width:3px; height:30px; border-radius:2px; background:var(--sev); }
.prio .t { font-weight:600; font-size:.88rem; }
.rules-mini { display:grid; gap:2px; }
.rules-mini > div { display:grid; grid-template-columns:30px 1fr auto; gap:10px; align-items:center; padding:5px 0; }
.rules-mini b { font-family:var(--mono); font-size:.8rem; color:var(--text-2); }

/* ---- incidents inbox */
.toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
.seg { display:inline-flex; border:1px solid var(--line-2); border-radius:10px; overflow:hidden; background:var(--surface); padding:2px; gap:2px; }
.seg button { border:none; background:none; padding:5px 11px; color:var(--text-2); cursor:pointer; font-size:.8rem; border-radius:8px; }
.seg button:hover { color:var(--text); background:var(--surface-2); }
.seg button[aria-pressed="true"] { background:var(--accent-soft); color:var(--accent); font-weight:600; }
.inbox { display:grid; grid-template-columns:minmax(280px, 340px) minmax(0,1fr); gap:16px; align-items:start; }
.list { max-height:calc(100vh - 210px); overflow:auto; padding:6px; position:sticky; top:80px; }
.row { display:grid; grid-template-columns:3px 1fr; gap:11px; padding:8px 10px; border-radius:9px; cursor:pointer;
  border:1px solid transparent; transition:background .12s; }
.row + .row { margin-top:1px; }
.row:hover { background:var(--surface-2); }
.row[aria-selected="true"] { background:var(--accent-soft); border-color:color-mix(in srgb, var(--accent) 55%, transparent); }
.row .bar { border-radius:2px; background:var(--sev); }
.row .top { display:flex; gap:8px; align-items:baseline; }
.row .t { font-weight:600; font-size:.87rem; }
.row .tm { margin-left:auto; font-family:var(--mono); font-size:.76rem; color:var(--text-2); }
.row .m { color:var(--text-3); font-size:.76rem; display:flex; gap:10px; flex-wrap:wrap; margin-top:1px; align-items:center; }
.row .verdict { font-size:.7rem; color:var(--text-3); }
.row[hidden], .detail[hidden], .rule-head[hidden] { display:none; }
.list .section-label { color:var(--text-3); font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; padding:12px 10px 4px; }
.list .rule-head { display:flex; align-items:center; gap:8px; padding:10px 10px 4px; font-size:.8rem; font-weight:600; color:var(--text); }
.list .rule-head b { font-family:var(--mono); font-size:.74rem; color:var(--text-2); background:var(--surface-2);
  border:1px solid var(--line); border-radius:6px; padding:0 5px; }
.list .rule-head span { margin-left:auto; color:var(--text-3); font-family:var(--mono); font-size:.74rem; font-weight:500; }
.detail { position:sticky; top:80px; }
.detail .d-head { padding:18px 20px 14px; border-bottom:1px solid var(--line); }
.detail .d-head h2 { margin:10px 0 2px; font-size:1.25rem; font-weight:700; }
.detail .d-head .when { font-family:var(--mono); font-size:.8rem; color:var(--text-2); }
.detail .d-body { padding:18px 20px; display:grid; grid-template-columns:minmax(0,1fr) 210px; gap:20px; }
.detail .media video, .detail .media img { width:100%; border-radius:10px; background:#000; display:block; }
.detail .crop img { width:100%; max-height:300px; object-fit:contain; border-radius:10px; background:var(--surface-2); }
.detail .say { margin:14px 0 0; font-size:.94rem; }
.detail .action { margin-top:12px; padding:11px 14px; border-radius:10px; background:var(--surface-2); font-size:.9rem;
  border-left:3px solid var(--sev); }
.detail .action b { display:block; font-size:.68rem; color:var(--text-3); text-transform:uppercase; letter-spacing:.08em; margin-bottom:2px; }
.facts { margin-top:14px; }
.facts td:first-child { color:var(--text-3); width:120px; }
.badge { font-size:.72rem; color:var(--text-3); margin-left:6px; }
.q { font-size:.72rem; color:var(--text-3); white-space:nowrap; }
.q::before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:5px; vertical-align:1px; background:var(--q); }
.btn.sm { padding:3px 10px; font-size:.78rem; border-radius:8px; }
figure.site { margin:0; }
figure.site img { width:100%; border-radius:10px; display:block; }
.review { display:flex; gap:6px; flex-wrap:wrap; align-items:center; padding:14px 20px; border-top:1px solid var(--line); }
.review .btn[aria-pressed="true"] { background:var(--accent); color:#fff; border-color:var(--accent); }
.trace svg { width:100%; height:84px; display:block; }
.trace .cap { color:var(--text-3); font-size:.78rem; margin-top:4px; }
.section-title { font-size:.7rem; font-weight:600; color:var(--text-3); text-transform:uppercase; letter-spacing:.08em; margin:20px 0 6px; }

/* ---- charts */
.bars .row-b { display:grid; grid-template-columns:150px minmax(0,1fr) 190px; gap:12px; align-items:center; margin:9px 0; }
.bars .track { height:12px; background:var(--surface-2); border-radius:4px; }
.bars .fill { height:12px; background:var(--series); border-radius:4px; min-width:2px; }
.bars .val { color:var(--text-2); font-size:.8rem; font-variant-numeric:tabular-nums; font-family:var(--mono); }
.stack { display:flex; gap:2px; height:10px; }
.stack span { display:block; height:100%; }
.stack span:first-child { border-radius:4px 0 0 4px; } .stack span:last-child { border-radius:0 4px 4px 0; }
.stack span:only-child { border-radius:4px; }
.cov-row { display:grid; grid-template-columns:40px 210px 150px minmax(0,1fr); gap:12px; align-items:center; padding:10px 0;
  border-bottom:1px solid var(--line); }
.cov-row > b { font-family:var(--mono); font-size:.82rem; color:var(--text-2); }
.cov-row:last-of-type { border-bottom:none; }
figure.heat { margin:0; }
figure.heat img { width:100%; border-radius:10px; display:block; }
figure.heat figcaption { color:var(--text-3); font-size:.8rem; margin-top:8px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.ramp { display:inline-block; width:110px; height:10px; border-radius:3px; background:linear-gradient(90deg, #faba76, #d03b3b); }
.tooltip { position:fixed; pointer-events:none; background:var(--surface); color:var(--text); border:1px solid var(--line-2);
  border-radius:9px; padding:8px 10px; font-size:.8rem; box-shadow:var(--shadow-lg); display:none; max-width:280px; z-index:60; }
ul.clean { margin:0; padding-left:18px; color:var(--text-2); }
ul.clean li + li { margin-top:5px; }

/* ---- assistant drawer */
.scrim { position:fixed; inset:0; background:rgba(5,8,12,.42); opacity:0; pointer-events:none; transition:opacity .2s; z-index:44; }
body.chat-open .scrim { opacity:1; pointer-events:auto; }
.fab { position:fixed; right:22px; bottom:22px; z-index:43; display:inline-flex; align-items:center; gap:8px; border:none;
  background:var(--accent); color:#fff; border-radius:999px; padding:12px 18px; font-weight:600; cursor:pointer;
  box-shadow:0 10px 30px color-mix(in srgb, var(--accent) 40%, transparent); transition:transform .15s; }
.fab:hover { transform:translateY(-2px); }
body.chat-open .fab { display:none; }

/* ---- responsive */
@media (max-width:1100px) {
  .inbox { grid-template-columns:1fr; }
  .detail, .list { position:static; max-height:none; }
  .grid.cols-3-2, .grid.cols-2 { grid-template-columns:1fr; }
}
@media (max-width:860px) {
  .shell { grid-template-columns:minmax(0,1fr); }
  .topbar .back { display:none; }
  .kpis { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .sidebar { position:sticky; top:60px; height:auto; z-index:20; padding:8px; border-right:none; border-bottom:1px solid var(--line);
    overflow-x:auto; flex-direction:row; }
  .sidebar .group, .sidebar .ask-side { display:none; }
  .nav { display:flex; gap:4px; }
  .nav a { white-space:nowrap; margin:0; }
  .nav a[aria-current="page"]::before { display:none; }
  main { padding:20px 16px 40px; }
  .hero { padding:20px; } .hero h1 { font-size:1.35rem; }
  .detail .d-body { grid-template-columns:1fr; }
  .cov-row { grid-template-columns:40px 1fr; } .cov-row > .stack, .cov-row > .chip-cell { grid-column:1 / -1; }
  .bars .row-b { grid-template-columns:1fr; }
  .topbar { padding:0 12px; gap:10px; }
  .topbar .meta, .topbar .divider, .topbar .wordmark, .topbar .hide-sm { display:none; }
}
"""

SCRIPT = """
(() => {
const $ = (q, el = document) => el.querySelector(q);
const $$ = (q, el = document) => Array.from(el.querySelectorAll(q));
const runId = document.body.dataset.run;

// ---- sections (hash routing)
function show(id) {
  const target = $('#page-' + id) ? id : 'overview';
  $$('section.page').forEach(s => s.classList.toggle('active', s.id === 'page-' + target));
  $$('.nav a').forEach(a => a.setAttribute('aria-current', a.dataset.page === target ? 'page' : 'false'));
  window.scrollTo({ top: 0 });
}
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1) || 'overview');

// ---- replay
const video = $('#video');
const playhead = $('#playhead');
const timeline = $('#timeline');
const duration = timeline ? Number(timeline.dataset.duration) : 1;
function seek(t) {
  if (!video) return;
  if (location.hash !== '#overview') location.hash = 'overview';
  video.currentTime = Math.max(0, t - 1);
  video.play().catch(() => {});
}
$$('[data-seek]').forEach(el => {
  el.addEventListener('click', e => { e.stopPropagation(); seek(Number(el.dataset.seek)); });
  el.addEventListener('keydown', e => { if (e.key === 'Enter') seek(Number(el.dataset.seek)); });
});
if (video && playhead) video.addEventListener('timeupdate', () => {
  const x = 60 + 920 * Math.min(1, video.currentTime / duration);
  playhead.setAttribute('x1', x); playhead.setAttribute('x2', x);
});

// ---- tooltips
const tip = $('#tip');
$$('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.innerHTML = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 300) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});

// ---- incidents inbox
const rows = $$('.row[data-id]');
function select(id) {
  rows.forEach(r => r.setAttribute('aria-selected', r.dataset.id === id ? 'true' : 'false'));
  $$('.detail[data-id]').forEach(d => {
    const on = d.dataset.id === id;
    d.hidden = !on;
    if (!on) { const v = $('video', d); if (v) v.pause(); }
  });
}
rows.forEach(r => r.addEventListener('click', () => select(r.dataset.id)));
window.openIncident = id => { location.hash = 'incidents'; select(id); };
$$('[data-open]').forEach(el => el.addEventListener('click', () => { location.hash = 'incidents'; select(el.dataset.open); }));
let ruleFilter = 'all', groupFilter = 'all';
function filter() {
  let first = null;
  rows.forEach(r => {
    const ok = (ruleFilter === 'all' || r.dataset.rule === ruleFilter) && (groupFilter === 'all' || r.dataset.group === groupFilter);
    r.hidden = !ok; if (ok && !first) first = r;
  });
  $$('.list .section-label').forEach(l => {
    l.hidden = !rows.some(r => !r.hidden && r.dataset.group === l.dataset.group);
  });
  $$('.list .rule-head').forEach(h => {
    h.hidden = !rows.some(r => !r.hidden && r.dataset.group === h.dataset.group && r.dataset.rule === h.dataset.rule);
  });
  const current = rows.find(r => r.getAttribute('aria-selected') === 'true' && !r.hidden);
  if (!current && first) select(first.dataset.id);
}
$$('[data-rule-filter]').forEach(b => b.addEventListener('click', () => {
  ruleFilter = b.dataset.ruleFilter;
  $$('[data-rule-filter]').forEach(x => x.setAttribute('aria-pressed', x === b));
  filter();
}));
$$('[data-group-filter]').forEach(b => b.addEventListener('click', () => {
  groupFilter = b.dataset.groupFilter;
  $$('[data-group-filter]').forEach(x => x.setAttribute('aria-pressed', x === b));
  filter();
}));
if (rows.length) select(rows[0].dataset.id);

// ---- human review (stored in this browser; exported for tools/review_incidents.py)
const key = 'safety-review:' + runId;
const reviewApi = document.querySelector('meta[name="safety-api"]');  // set when served by app/cloud.py
let verdicts = {};
try { verdicts = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { verdicts = {}; }
function paint() {
  rows.forEach(r => {
    const v = verdicts[r.dataset.id];
    const tag = $('.verdict', r);
    if (tag) tag.textContent = v ? ({real: '✓ real', false_alarm: '✗ false alarm', unsure: '? unsure'})[v.verdict] : '';
  });
  $$('.review').forEach(box => {
    const v = verdicts[box.dataset.id];
    $$('[data-verdict]', box).forEach(b => b.setAttribute('aria-pressed', v && v.verdict === b.dataset.verdict ? 'true' : 'false'));
  });
  const done = Object.keys(verdicts).length;
  $$('[data-review-count]').forEach(el => el.textContent = done + ' of ' + rows.length + ' reviewed');
}
$$('.review').forEach(box => $$('[data-verdict]', box).forEach(b => b.addEventListener('click', () => {
  const entry = { verdict: b.dataset.verdict };
  if (b.dataset.reason) entry.reason = b.dataset.reason;
  verdicts[box.dataset.id] = entry;
  try { localStorage.setItem(key, JSON.stringify(verdicts)); } catch (e) {}
  if (reviewApi) {  // hosted: also store centrally, under the reviewer's name
    const name = reviewer();
    if (name) fetch(reviewApi.content + '/reviews', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ incident_id: box.dataset.id, ...entry, reviewer: name }) })
      .then(() => setTimeout(team, 1500)).catch(() => {});
  }
  paint();
})));
function reviewer() {
  let name = '';
  try { name = localStorage.getItem('safety-reviewer') || ''; } catch (e) {}
  if (!name) {
    name = (window.prompt('Your name, so your reviews count in the team precision:') || '')
      .replace(/[^A-Za-z0-9_ .'-]/g, '').trim().slice(0, 40);
    if (name) { try { localStorage.setItem('safety-reviewer', name); } catch (e) {} }
  }
  return name;
}
function team() {  // team precision and agreement, from every reviewer's verdicts
  const el = $('#team-review');
  if (!reviewApi || !el) return;
  fetch(reviewApi.content + '/reviews').then(r => r.ok ? r.json() : null).then(s => {
    if (!s || !s.reviewers) return;
    const k = s.agreement.mean_kappa;
    el.textContent = 'Team: ' + s.reviewers + (s.reviewers > 1 ? ' reviewers' : ' reviewer') + ', '
      + s.decided + ' decided, precision ' + (s.precision == null ? 'n/a' : Math.round(100 * s.precision) + '%')
      + (k == null ? '' : ', agreement kappa ' + k.toFixed(2));
    el.hidden = false;
  }).catch(() => {});
}
team();
const exportButton = $('#export-review');
if (exportButton) exportButton.addEventListener('click', () => {
  const lines = Object.entries(verdicts).map(([id, v]) => JSON.stringify({ run_id: runId, incident_id: id, ...v })).join('\\n');
  const blob = new Blob([lines + '\\n'], { type: 'application/x-ndjson' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'incident_verdicts_' + runId + '.jsonl'; a.click();
});
paint();

// ---- hosted: a way back to every run
if (reviewApi) $$('[data-hosted]').forEach(el => { el.hidden = false; });

// ---- assistant drawer, reachable from every section
function chat(open) {
  document.body.classList.toggle('chat-open', open);
  const panel = $('#chat');
  if (panel) panel.setAttribute('aria-hidden', open ? 'false' : 'true');
  if (open) { const q = $('#chat-q'); if (q) setTimeout(() => q.focus(), 180); }
}
$$('[data-ask-open]').forEach(b => b.addEventListener('click', () => chat(true)));
$$('[data-ask-close], .scrim').forEach(b => b.addEventListener('click', () => chat(false)));
document.addEventListener('keydown', e => { if (e.key === 'Escape') chat(false); });
window.closeChat = () => { if (innerWidth < 1100) chat(false); };

// ---- theme
const toggle = $('#theme');
if (toggle) toggle.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('safety-theme', root.dataset.theme); } catch (e) {}
});
try { const saved = localStorage.getItem('safety-theme'); if (saved) document.documentElement.dataset.theme = saved; } catch (e) {}
})();
"""

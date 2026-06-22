"""
Enrichment Data Dashboard
Browse and explore enrichment data collected via the enrichment pipeline.
Shows per-professor enrichment results from 13 public sources.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profiles", tags=["profiles"])

PROFILES_DIR = Path(__file__).parent.parent.parent / "output" / "osu_faculty_run" / "profiles"
CHUNKS_DIR = Path(__file__).parent.parent.parent / "output" / "osu_faculty_run" / "chunked_profiles"

ALL_SOURCES = [
    "orcid", "osu_expertise", "semantic_scholar", "openalex", "crossref",
    "google_scholar", "nsf_grants", "nih_grants", "osu_courses",
    "rate_my_professor", "osu_news", "google_news", "youtube_lectures",
]

SOURCE_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "google_scholar": "Google Scholar",
    "crossref": "CrossRef",
    "orcid": "ORCID",
    "nsf_grants": "NSF Grants",
    "nih_grants": "NIH Grants",
    "rate_my_professor": "RateMyProfessors",
    "osu_courses": "OSU Courses",
    "osu_news": "OSU News",
    "google_news": "Google News",
    "youtube_lectures": "YouTube Lectures",
    "osu_expertise": "OSU Expertise",
}

SOURCE_ICONS = {
    "semantic_scholar": "S2",
    "openalex": "OA",
    "google_scholar": "GS",
    "crossref": "CR",
    "orcid": "OR",
    "nsf_grants": "NSF",
    "nih_grants": "NIH",
    "rate_my_professor": "RMP",
    "osu_courses": "CRS",
    "osu_news": "ON",
    "google_news": "GN",
    "youtube_lectures": "YT",
    "osu_expertise": "EXP",
}


# ── JSON API ────────────────────────────────────────────────────────────

@router.get("/api/professors")
async def list_professors(
    enriched_only: bool = False,
    search: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    """List professors with enrichment status summary."""
    if not PROFILES_DIR.exists():
        return {"professors": [], "total": 0, "enriched_count": 0}

    professors = []
    enriched_count = 0

    for pdir in sorted(PROFILES_DIR.iterdir()):
        if not pdir.is_dir():
            continue
        profile_json = pdir / f"{pdir.name}.json"
        if not profile_json.exists():
            continue
        try:
            data = json.loads(profile_json.read_text(encoding="utf-8"))
        except Exception:
            continue

        name = data.get("name", "")
        if not name or len(name.strip()) < 3:
            continue

        if search and search.lower() not in name.lower():
            continue

        enrichment_json = pdir / "enrichment.json"
        has_enrichment = enrichment_json.exists()
        source_count = 0
        failed_count = 0
        confidence = 0.0
        enriched_at = None

        if has_enrichment:
            enriched_count += 1
            try:
                enr = json.loads(enrichment_json.read_text(encoding="utf-8"))
                source_count = enr.get("summary", {}).get("successful_sources", 0)
                failed_count = enr.get("summary", {}).get("failed_sources", 0)
                confidence = enr.get("confidence", {}).get("overall_confidence", 0.0)
                enriched_at = enr.get("enriched_at")
            except Exception:
                pass

        if enriched_only and not has_enrichment:
            continue

        has_chunks = (CHUNKS_DIR / pdir.name / "chunks.json").exists()
        has_text = (pdir / "enrichment_text.txt").exists()

        professors.append({
            "profile_id": pdir.name,
            "name": name,
            "profile_url": data.get("profile_url", ""),
            "has_enrichment": has_enrichment,
            "source_count": source_count,
            "failed_count": failed_count,
            "confidence": round(confidence, 3),
            "enriched_at": enriched_at,
            "has_chunks": has_chunks,
            "has_text": has_text,
        })

    professors.sort(key=lambda p: (-p["source_count"], -p["confidence"], p["name"].lower()))
    total = len(professors)
    page = professors[offset:offset + limit]

    return {"professors": page, "total": total, "enriched_count": enriched_count}


@router.get("/api/professor/{profile_id}")
async def get_professor_detail(profile_id: str):
    """Full enrichment detail for one professor."""
    pdir = PROFILES_DIR / profile_id
    if not pdir.is_dir():
        raise HTTPException(404, "Profile not found")

    profile_json = pdir / f"{profile_id}.json"
    if not profile_json.exists():
        raise HTTPException(404, "Profile JSON not found")

    try:
        profile_data = json.loads(profile_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Error reading profile: {e}")

    result = {
        "profile_id": profile_id,
        "name": profile_data.get("name", ""),
        "profile_url": profile_data.get("profile_url", ""),
        "has_enrichment": False,
        "enrichment": None,
        "enrichment_text": None,
        "chunks": None,
    }

    # Enrichment JSON
    enrichment_json = pdir / "enrichment.json"
    if enrichment_json.exists():
        try:
            result["enrichment"] = json.loads(enrichment_json.read_text(encoding="utf-8"))
            result["has_enrichment"] = True
        except Exception:
            pass

    # Enrichment text
    enrichment_txt = pdir / "enrichment_text.txt"
    if enrichment_txt.exists():
        try:
            result["enrichment_text"] = enrichment_txt.read_text(encoding="utf-8")
        except Exception:
            pass

    # Chunks
    chunks_json = CHUNKS_DIR / profile_id / "chunks.json"
    if chunks_json.exists():
        try:
            result["chunks"] = json.loads(chunks_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    return result


# ── HTML UI ─────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def enrichment_dashboard():
    """Serve the enrichment data dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@router.get("/{profile_id}", response_class=HTMLResponse)
async def professor_detail_page(profile_id: str):
    """Serve detail page for a single professor."""
    return HTMLResponse(content=_DETAIL_HTML)


# ── Dashboard HTML ──────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enrichment Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0b0e17;min-height:100vh;color:#c8cdd5}
.wrap{max-width:1400px;margin:0 auto;padding:20px}
.header{text-align:center;padding:32px 20px 24px;border-bottom:1px solid #1e2536}
.header h1{font-size:2.2em;color:#60a5fa;letter-spacing:-0.5px}
.header p{color:#6b7280;margin-top:6px}
.stats-row{display:flex;gap:14px;flex-wrap:wrap;padding:20px 0;justify-content:center}
.stat{background:#111827;border:1px solid #1e2536;border-radius:10px;padding:16px 24px;text-align:center;min-width:150px}
.stat .n{font-size:1.9em;font-weight:700;color:#60a5fa}
.stat .l{color:#6b7280;font-size:.85em;margin-top:2px}
.controls{display:flex;gap:12px;flex-wrap:wrap;padding:16px 0;align-items:center}
.controls input,.controls select{padding:10px 14px;border:1px solid #1e2536;border-radius:8px;font-size:14px;background:#111827;color:#c8cdd5}
.controls input{flex:1;min-width:240px}
.controls input:focus,.controls select:focus{outline:none;border-color:#60a5fa}
.controls select{min-width:180px}
#jumpTo{flex:1.5;min-width:300px;color:#93c5fd}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px;padding-bottom:40px}
.card{background:#111827;border:1px solid #1e2536;border-radius:10px;padding:18px 20px;cursor:pointer;transition:border-color .2s,transform .15s}
.card:hover{border-color:#60a5fa;transform:translateY(-2px)}
.card h3{color:#e2e8f0;font-size:1.1em;margin-bottom:8px}
.card .url{color:#6b7280;font-size:.8em;margin-bottom:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.card .url a{color:#60a5fa;text-decoration:none}
.tag-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:.75em;font-weight:600}
.tag-ok{background:#064e3b;color:#6ee7b7}
.tag-fail{background:#7f1d1d;color:#fca5a5}
.tag-chunks{background:#1e3a5f;color:#93c5fd}
.tag-none{background:#1f2937;color:#6b7280}
.conf-bar{height:5px;border-radius:3px;background:#1f2937;overflow:hidden;margin-top:10px}
.conf-fill{height:100%;border-radius:3px;transition:width .4s}
.card .meta{color:#6b7280;font-size:.8em;margin-top:6px}
.empty{text-align:center;padding:60px 20px;color:#4b5563}
.page-controls{display:flex;justify-content:center;gap:12px;padding:20px 0}
.page-controls button{padding:8px 18px;border:1px solid #1e2536;border-radius:8px;background:#111827;color:#c8cdd5;cursor:pointer;font-size:.9em}
.page-controls button:hover{border-color:#60a5fa}
.page-controls button:disabled{opacity:.4;cursor:default}
.page-controls span{color:#6b7280;padding:8px 0;font-size:.9em}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>Enrichment Pipeline Dashboard</h1>
    <p>Faculty profiles enriched from 13 public data sources</p>
  </div>
  <div class="stats-row" id="statsRow"></div>
  <div class="controls">
    <select id="jumpTo">
      <option value="">-- Jump to enriched professor --</option>
    </select>
    <input type="text" id="search" placeholder="Search by name...">
    <select id="filter">
      <option value="all">All Professors</option>
      <option value="enriched" selected>Enriched Only</option>
    </select>
  </div>
  <div id="grid" class="grid"></div>
  <div class="page-controls" id="pageControls"></div>
</div>
<script>
const PAGE_SIZE=60;
let currentPage=0,totalProfs=0;

async function loadEnrichedDropdown(){
  const r=await fetch('/profiles/api/professors?enriched_only=true&limit=5000');
  const d=await r.json();
  const sel=document.getElementById('jumpTo');
  sel.innerHTML='<option value="">-- Jump to enriched professor ('+d.professors.length+') --</option>';
  d.professors.forEach(p=>{
    const opt=document.createElement('option');
    opt.value=p.profile_id;
    opt.textContent=p.name+' ['+p.source_count+' sources, '+(p.confidence*100).toFixed(0)+'%]';
    sel.appendChild(opt);
  });
}
document.getElementById('jumpTo').addEventListener('change',e=>{
  if(e.target.value)window.location.href='/profiles/'+e.target.value;
});

async function load(){
  const q=document.getElementById('search').value;
  const f=document.getElementById('filter').value;
  const params=new URLSearchParams({limit:PAGE_SIZE,offset:currentPage*PAGE_SIZE});
  if(q)params.set('search',q);
  if(f==='enriched')params.set('enriched_only','true');
  const r=await fetch('/profiles/api/professors?'+params);
  const d=await r.json();
  totalProfs=d.total;
  renderStats(d);
  renderGrid(d.professors);
  renderPaging();
}

function renderStats(d){
  document.getElementById('statsRow').innerHTML=
    st(d.total,'Total Professors')+st(d.enriched_count,'Enriched')+
    st(d.total-d.enriched_count,'Pending');
}
function st(n,l){return '<div class="stat"><div class="n">'+n+'</div><div class="l">'+l+'</div></div>'}

function renderGrid(list){
  const g=document.getElementById('grid');
  if(!list.length){g.innerHTML='<div class="empty"><h3>No professors found</h3></div>';return}
  g.innerHTML=list.map(p=>{
    const confPct=(p.confidence*100).toFixed(0);
    const confColor=p.confidence>.7?'#10b981':p.confidence>.4?'#f59e0b':'#ef4444';
    let tags='';
    if(p.has_enrichment){
      tags+='<span class="tag tag-ok">'+p.source_count+' sources</span>';
      if(p.failed_count)tags+='<span class="tag tag-fail">'+p.failed_count+' failed</span>';
      if(p.has_chunks)tags+='<span class="tag tag-chunks">chunked</span>';
    } else {
      tags+='<span class="tag tag-none">not enriched</span>';
    }
    let meta='';
    if(p.enriched_at){
      const d=new Date(p.enriched_at);
      meta='Enriched '+d.toLocaleDateString()+' &middot; Confidence '+confPct+'%';
    }
    let url='';
    if(p.profile_url)url='<div class="url"><a href="'+esc(p.profile_url)+'" target="_blank" onclick="event.stopPropagation()">'+esc(p.profile_url)+'</a></div>';
    let bar='';
    if(p.has_enrichment)bar='<div class="conf-bar"><div class="conf-fill" style="width:'+confPct+'%;background:'+confColor+'"></div></div>';
    return '<div class="card" onclick="go(&quot;'+p.profile_id+'&quot;)"><h3>'+esc(p.name)+'</h3>'+url+'<div class="tag-row">'+tags+'</div>'+bar+'<div class="meta">'+meta+'</div></div>';
  }).join('');
}

function renderPaging(){
  const pages=Math.ceil(totalProfs/PAGE_SIZE);
  const c=document.getElementById('pageControls');
  if(pages<=1){c.innerHTML='';return}
  c.innerHTML='<button '+(currentPage===0?'disabled':'')+' onclick="prev()">Prev</button>'+
    '<span>Page '+(currentPage+1)+' of '+pages+'</span>'+
    '<button '+(currentPage>=pages-1?'disabled':'')+' onclick="next()">Next</button>';
}
function prev(){if(currentPage>0){currentPage--;load()}}
function next(){currentPage++;load()}
function go(id){window.location.href='/profiles/'+id}
function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML}

let debounce;
document.getElementById('search').addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(()=>{currentPage=0;load()},300)});
document.getElementById('filter').addEventListener('change',()=>{currentPage=0;load()});
load();
loadEnrichedDropdown();
</script>
</body>
</html>"""


# ── Detail Page HTML ────────────────────────────────────────────────────

_DETAIL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Professor Enrichment Detail</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0b0e17;min-height:100vh;color:#c8cdd5}
.wrap{max-width:1200px;margin:0 auto;padding:20px}
a.back{color:#60a5fa;text-decoration:none;font-size:.95em;display:inline-block;margin-bottom:16px}
a.back:hover{text-decoration:underline}
.prof-card{background:#111827;border:1px solid #1e2536;border-radius:12px;padding:24px;margin-bottom:20px;border-left:4px solid #60a5fa}
.prof-card h1{color:#e2e8f0;font-size:1.8em;margin-bottom:6px}
.prof-card .meta{color:#6b7280;font-size:.9em;margin:3px 0}
.prof-card .meta a{color:#60a5fa;text-decoration:none}
.stats-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}
.stat{background:#111827;border:1px solid #1e2536;border-radius:10px;padding:14px 20px;flex:1;min-width:130px;text-align:center}
.stat .n{font-size:1.7em;font-weight:700;color:#60a5fa}
.stat .l{color:#6b7280;font-size:.82em;margin-top:2px}
.conf-bar{height:6px;border-radius:3px;background:#1f2937;overflow:hidden;margin-bottom:20px}
.conf-fill{height:100%;border-radius:3px;transition:width .5s}
.tabs{display:flex;gap:0;border-bottom:2px solid #1e2536;margin-bottom:20px}
.tab{padding:12px 24px;cursor:pointer;color:#6b7280;font-weight:600;font-size:.95em;border-bottom:2px solid transparent;margin-bottom:-2px;transition:color .2s,border-color .2s}
.tab:hover{color:#c8cdd5}
.tab.active{color:#60a5fa;border-bottom-color:#60a5fa}
.panel{display:none}
.panel.active{display:block}

/* Sources tab */
.sources-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.src-card{background:#111827;border:1px solid #1e2536;border-radius:10px;overflow:hidden;transition:border-color .2s}
.src-card:hover{border-color:#374151}
.src-hdr{padding:14px 18px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;border-bottom:1px solid #1e2536}
.src-hdr h3{font-size:1em;display:flex;align-items:center;gap:8px}
.src-icon{width:30px;height:30px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:.7em;font-weight:700;color:#fff;flex-shrink:0}
.src-ok{color:#6ee7b7;font-weight:600;font-size:.85em}
.src-fail{color:#fca5a5;font-weight:600;font-size:.85em}
.src-cached{color:#fbbf24;font-size:.75em;margin-left:4px}
.src-body{max-height:500px;overflow-y:auto;padding:16px 18px;display:none;font-size:.9em;line-height:1.6}
.src-body.open{display:block}
.src-body pre{background:#0d1117;color:#c9d1d9;padding:14px;border-radius:8px;overflow-x:auto;font-size:.82em;white-space:pre-wrap;word-break:break-word}

/* Key metrics */
.metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:16px}
.metric{background:#0d1117;border-radius:8px;padding:12px 16px}
.metric .k{color:#6b7280;font-size:.8em;text-transform:uppercase;letter-spacing:.5px}
.metric .v{color:#e2e8f0;font-size:1.1em;font-weight:600;margin-top:2px}

/* Pubs table */
.pub-list{margin-top:12px}
.pub{background:#0d1117;border-radius:8px;padding:12px 16px;margin-bottom:8px}
.pub .title{color:#93c5fd;font-weight:600;font-size:.92em}
.pub .detail{color:#6b7280;font-size:.82em;margin-top:4px}

/* Text tab */
.enr-text{background:#111827;border:1px solid #1e2536;border-radius:10px;padding:24px;white-space:pre-wrap;font-family:'Consolas','Monaco',monospace;font-size:.88em;line-height:1.7;max-height:800px;overflow-y:auto;color:#9ca3af}

/* Chunks tab */
.chunk-section{margin-bottom:20px}
.chunk-section h3{color:#60a5fa;margin-bottom:10px;font-size:1.05em}
.chunk{background:#111827;border:1px solid #1e2536;border-radius:8px;padding:16px;margin-bottom:10px}
.chunk .text{color:#d1d5db;font-size:.9em;line-height:1.6;margin-bottom:8px}
.chunk .cmeta{color:#4b5563;font-size:.78em;display:flex;gap:12px;flex-wrap:wrap}

.empty{text-align:center;padding:50px 20px;color:#4b5563}
.loading{text-align:center;padding:40px;color:#6b7280}
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="/profiles/">&larr; Back to Dashboard</a>
  <div id="content"><div class="loading">Loading...</div></div>
</div>
<script>
const SOURCE_LABELS={"semantic_scholar":"Semantic Scholar","openalex":"OpenAlex","google_scholar":"Google Scholar","crossref":"CrossRef","orcid":"ORCID","nsf_grants":"NSF Grants","nih_grants":"NIH Grants","rate_my_professor":"RateMyProfessors","osu_courses":"OSU Courses","osu_news":"OSU News","google_news":"Google News","youtube_lectures":"YouTube Lectures","osu_expertise":"OSU Expertise"};
const SOURCE_ICONS={"semantic_scholar":"S2","openalex":"OA","google_scholar":"GS","crossref":"CR","orcid":"OR","nsf_grants":"NSF","nih_grants":"NIH","rate_my_professor":"RMP","osu_courses":"CRS","osu_news":"ON","google_news":"GN","youtube_lectures":"YT","osu_expertise":"EXP"};
const ICON_COLORS={"semantic_scholar":"#2563eb","openalex":"#c026d3","google_scholar":"#4285f4","crossref":"#ea580c","orcid":"#a3c51c","nsf_grants":"#0369a1","nih_grants":"#0d9488","rate_my_professor":"#facc15","osu_courses":"#dc2626","osu_news":"#dc2626","google_news":"#ea4335","youtube_lectures":"#ff0000","osu_expertise":"#bb1632"};
const ALL_SOURCES=["orcid","osu_expertise","semantic_scholar","openalex","crossref","google_scholar","nsf_grants","nih_grants","osu_courses","rate_my_professor","osu_news","google_news","youtube_lectures"];

const pid=window.location.pathname.split('/').filter(Boolean).pop();

async function load(){
  try{
    const r=await fetch('/profiles/api/professor/'+pid);
    if(!r.ok)throw new Error('Not found');
    const d=await r.json();
    render(d);
  }catch(e){
    document.getElementById('content').innerHTML='<div class="empty"><h3>Error loading profile</h3><p>'+esc(e.message)+'</p></div>';
  }
}

function render(d){
  const c=document.getElementById('content');
  let h='';

  // Prof card
  h+='<div class="prof-card"><h1>'+esc(d.name)+'</h1>';
  if(d.profile_url)h+='<div class="meta"><a href="'+esc(d.profile_url)+'" target="_blank">'+esc(d.profile_url)+'</a></div>';
  h+='<div class="meta">Profile ID: '+esc(d.profile_id)+'</div></div>';

  if(!d.has_enrichment||!d.enrichment){
    h+='<div class="empty"><h3>No enrichment data yet</h3><p>Run: <code>python run_enrichment_pipeline.py --name "'+esc(d.name)+'"</code></p></div>';
    c.innerHTML=h;return;
  }

  const enr=d.enrichment;
  const sum=enr.summary||{};
  const conf=enr.confidence||{};
  const cs=conf.overall_confidence||0;

  // Stats
  h+='<div class="stats-row">';
  h+=stat(sum.successful_sources||0,'Sources OK');
  h+=stat(sum.failed_sources||0,'Failed');
  h+=stat((cs*100).toFixed(0)+'%','Confidence');
  h+=stat(conf.name_match_sources||0,'Name Matches');
  h+=stat(conf.osu_affiliation_confirmed||0,'OSU Confirmed');
  const hi=conf.h_index_values||[];
  if(hi.length)h+=stat(Math.max(...hi),'h-index');
  const cc=conf.citation_counts||[];
  if(cc.length)h+=stat(Math.max(...cc).toLocaleString(),'Citations');
  h+='</div>';

  // Confidence bar
  const clr=cs>.7?'#10b981':cs>.4?'#f59e0b':'#ef4444';
  h+='<div class="conf-bar"><div class="conf-fill" style="width:'+(cs*100)+'%;background:'+clr+'"></div></div>';

  // Tabs
  h+='<div class="tabs">';
  h+='<div class="tab active" data-tab="sources">Sources</div>';
  if(d.enrichment_text)h+='<div class="tab" data-tab="text">Enrichment Text</div>';
  if(d.chunks)h+='<div class="tab" data-tab="chunks">Chunks</div>';
  h+='</div>';

  // Sources panel
  h+='<div class="panel active" id="panel-sources"><div class="sources-grid">';
  ALL_SOURCES.forEach(s=>{
    const sd=(enr.sources||{})[s];
    h+=renderSource(s,sd);
  });
  h+='</div></div>';

  // Text panel
  if(d.enrichment_text){
    h+='<div class="panel" id="panel-text"><div class="enr-text">'+esc(d.enrichment_text)+'</div></div>';
  }

  // Chunks panel
  if(d.chunks){
    h+='<div class="panel" id="panel-chunks">'+renderChunks(d.chunks)+'</div>';
  }

  c.innerHTML=h;

  // Tab switching
  document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-'+t.dataset.tab).classList.add('active');
  }));

  // Source card toggles
  document.querySelectorAll('.src-hdr').forEach(hdr=>hdr.addEventListener('click',()=>{
    hdr.nextElementSibling.classList.toggle('open');
  }));
}

function renderSource(name,sd){
  const label=SOURCE_LABELS[name]||name;
  const icon=SOURCE_ICONS[name]||'?';
  const color=ICON_COLORS[name]||'#374151';
  let h='<div class="src-card"><div class="src-hdr"><h3><span class="src-icon" style="background:'+color+'">'+icon+'</span>'+esc(label)+'</h3>';
  if(!sd){
    h+='<span class="src-fail">Not queried</span></div>';
    h+='<div class="src-body"><p style="color:#4b5563">Not queried during enrichment.</p></div></div>';
    return h;
  }
  if(sd.success){
    h+='<span class="src-ok">OK</span>';
    if(sd.cached)h+='<span class="src-cached">cached</span>';
  }else{
    h+='<span class="src-fail">Failed</span>';
  }
  h+='</div><div class="src-body">';
  if(sd.success&&sd.data&&Object.keys(sd.data).length){
    h+=renderSourceData(name,sd.data);
  }else if(sd.error){
    h+='<p style="color:#fca5a5">'+esc(sd.error)+'</p>';
  }else{
    h+='<p style="color:#4b5563">No data returned.</p>';
  }
  h+='</div></div>';
  return h;
}

function renderSourceData(name,data){
  // Smart rendering per source type
  let h='';
  if(name==='openalex'){
    h+='<div class="metrics">';
    if(data.works_count!=null)h+=metric('Works',data.works_count);
    if(data.cited_by_count!=null)h+=metric('Citations',data.cited_by_count.toLocaleString());
    if(data.h_index!=null)h+=metric('h-index',data.h_index);
    if(data.i10_index!=null)h+=metric('i10-index',data.i10_index);
    h+='</div>';
    if(data.concepts&&data.concepts.length){
      h+='<div style="margin-bottom:12px"><span style="color:#6b7280;font-size:.82em">Top concepts:</span><div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap">';
      data.concepts.slice(0,10).forEach(c=>{
        h+='<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:.78em">'+esc(c.display_name||c)+'</span>';
      });
      h+='</div></div>';
    }
    if(data.works&&data.works.length){
      h+='<div class="pub-list">';
      data.works.slice(0,10).forEach(w=>{
        h+='<div class="pub"><div class="title">'+esc(w.title||'Untitled')+'</div>';
        const parts=[];
        if(w.publication_year)parts.push(w.publication_year);
        if(w.cited_by_count!=null)parts.push(w.cited_by_count+' citations');
        if(w.doi)parts.push(w.doi);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      if(data.works.length>10)h+='<div style="color:#4b5563;font-size:.82em;padding:8px">...and '+(data.works.length-10)+' more</div>';
      h+='</div>';
    }
    return h;
  }

  if(name==='semantic_scholar'){
    h+='<div class="metrics">';
    if(data.h_index!=null)h+=metric('h-index',data.h_index);
    if(data.citation_count!=null)h+=metric('Citations',data.citation_count.toLocaleString());
    if(data.paper_count!=null)h+=metric('Papers',data.paper_count);
    h+='</div>';
    if(data.papers&&data.papers.length){
      h+='<div class="pub-list">';
      data.papers.slice(0,10).forEach(p=>{
        h+='<div class="pub"><div class="title">'+esc(p.title||'Untitled')+'</div>';
        const parts=[];
        if(p.year)parts.push(p.year);
        if(p.citationCount!=null)parts.push(p.citationCount+' citations');
        if(p.venue)parts.push(p.venue);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      if(data.papers.length>10)h+='<div style="color:#4b5563;font-size:.82em;padding:8px">...and '+(data.papers.length-10)+' more</div>';
      h+='</div>';
    }
    return h;
  }

  if(name==='orcid'){
    h+='<div class="metrics">';
    if(data.orcid_id)h+=metric('ORCID ID',data.orcid_id);
    h+='</div>';
    if(data.education&&data.education.length){
      h+='<div style="margin:10px 0"><strong style="color:#9ca3af;font-size:.85em">Education</strong>';
      data.education.forEach(e=>{
        h+='<div class="pub"><div class="title">'+esc(e.role_title||e.department||'')+'</div>';
        h+='<div class="detail">'+esc(e.organization||'')+' '+esc((e.start_year||'')+' - '+(e.end_year||'present'))+'</div></div>';
      });
      h+='</div>';
    }
    if(data.employment&&data.employment.length){
      h+='<div style="margin:10px 0"><strong style="color:#9ca3af;font-size:.85em">Employment</strong>';
      data.employment.forEach(e=>{
        h+='<div class="pub"><div class="title">'+esc(e.role_title||e.department||'')+'</div>';
        h+='<div class="detail">'+esc(e.organization||'')+' '+esc((e.start_year||'')+' - '+(e.end_year||'present'))+'</div></div>';
      });
      h+='</div>';
    }
    return h;
  }

  if(name==='rate_my_professor'){
    h+='<div class="metrics">';
    if(data.overall_rating!=null)h+=metric('Rating',data.overall_rating+' / 5');
    if(data.difficulty_rating!=null)h+=metric('Difficulty',data.difficulty_rating+' / 5');
    if(data.num_ratings!=null)h+=metric('Ratings',data.num_ratings);
    if(data.would_take_again!=null)h+=metric('Would Take Again',data.would_take_again+'%');
    h+='</div>';
    if(data.top_tags&&data.top_tags.length){
      h+='<div style="margin-bottom:12px;display:flex;gap:6px;flex-wrap:wrap">';
      data.top_tags.forEach(t=>{
        h+='<span style="background:#422006;color:#fbbf24;padding:3px 10px;border-radius:12px;font-size:.78em">'+esc(t)+'</span>';
      });
      h+='</div>';
    }
    if(data.reviews&&data.reviews.length){
      h+='<div class="pub-list">';
      data.reviews.slice(0,8).forEach(r=>{
        h+='<div class="pub"><div class="text" style="color:#d1d5db;font-size:.88em">"'+esc(r.comment||r.review||'')+'"</div>';
        const parts=[];
        if(r.quality!=null)parts.push('Quality: '+r.quality);
        if(r.difficulty!=null)parts.push('Difficulty: '+r.difficulty);
        if(r.course)parts.push(r.course);
        if(r.grade)parts.push('Grade: '+r.grade);
        if(r.date)parts.push(r.date);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      h+='</div>';
    }
    return h;
  }

  if(name==='nsf_grants'||name==='nih_grants'){
    const grants=data.grants||data.projects||[];
    if(grants.length){
      h+='<div class="metrics">';
      h+=metric('Grants Found',grants.length);
      const totalAmt=grants.reduce((s,g)=>s+(g.award_amount||g.amount||g.total_cost||0),0);
      if(totalAmt)h+=metric('Total Funding','$'+totalAmt.toLocaleString());
      h+='</div>';
      h+='<div class="pub-list">';
      grants.slice(0,8).forEach(g=>{
        h+='<div class="pub"><div class="title">'+esc(g.title||g.project_title||'Untitled')+'</div>';
        const parts=[];
        if(g.award_amount||g.amount||g.total_cost)parts.push('$'+(g.award_amount||g.amount||g.total_cost).toLocaleString());
        if(g.start_date)parts.push(g.start_date);
        if(g.agency)parts.push(g.agency);
        if(g.award_id||g.project_num)parts.push(g.award_id||g.project_num);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      h+='</div>';
    }
    return h||'<pre>'+esc(JSON.stringify(data,null,2))+'</pre>';
  }

  if(name==='crossref'){
    const works=data.works||data.publications||[];
    if(works.length){
      h+='<div class="metrics">';
      h+=metric('Publications',works.length);
      h+='</div><div class="pub-list">';
      works.slice(0,10).forEach(w=>{
        h+='<div class="pub"><div class="title">'+esc(w.title||'Untitled')+'</div>';
        const parts=[];
        if(w.year||w.published)parts.push(w.year||w.published);
        if(w.cited_by_count||w.is_referenced_by_count)parts.push((w.cited_by_count||w.is_referenced_by_count)+' citations');
        if(w.doi)parts.push(w.doi);
        if(w.container_title)parts.push(w.container_title);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      h+='</div>';
      return h;
    }
  }

  if(name==='osu_courses'){
    const courses=data.courses||[];
    if(courses.length){
      h+='<div class="metrics">'+metric('Courses',courses.length)+'</div>';
      h+='<div class="pub-list">';
      courses.forEach(cr=>{
        h+='<div class="pub"><div class="title">'+esc(cr.course_code||'')+' '+esc(cr.title||cr.name||'')+'</div>';
        if(cr.description)h+='<div class="detail">'+esc(cr.description.substring(0,200))+'</div>';
        h+='</div>';
      });
      h+='</div>';
      return h;
    }
  }

  if(name==='osu_news'||name==='google_news'){
    const articles=data.articles||data.news||[];
    if(articles.length){
      h+='<div class="metrics">'+metric('Articles',articles.length)+'</div>';
      h+='<div class="pub-list">';
      articles.slice(0,8).forEach(a=>{
        h+='<div class="pub"><div class="title">'+esc(a.title||'Untitled')+'</div>';
        const parts=[];
        if(a.date||a.published)parts.push(a.date||a.published);
        if(a.source)parts.push(a.source);
        if(parts.length)h+='<div class="detail">'+parts.join(' &middot; ')+'</div>';
        h+='</div>';
      });
      h+='</div>';
      return h;
    }
  }

  // Fallback: raw JSON
  h+='<pre>'+esc(JSON.stringify(data,null,2))+'</pre>';
  return h;
}

function renderChunks(chunks){
  const sections=chunks.sections||{};
  const keys=Object.keys(sections);
  if(!keys.length)return '<div class="empty">No chunks available</div>';
  let h='';
  keys.forEach(sec=>{
    h+='<div class="chunk-section"><h3>'+esc(sec)+' ('+sections[sec].length+')</h3>';
    sections[sec].forEach(ch=>{
      h+='<div class="chunk"><div class="text">'+esc(ch.text||ch.raw_text||'')+'</div>';
      h+='<div class="cmeta">';
      if(ch.source_id)h+='<span>Source: '+esc(ch.source_id)+'</span>';
      if(ch.allowed_use)h+='<span>Use: '+esc(ch.allowed_use)+'</span>';
      if(ch.quote_ok!=null)h+='<span>Quotable: '+(ch.quote_ok?'Yes':'No')+'</span>';
      if(ch.is_summary)h+='<span>Summary</span>';
      h+='</div></div>';
    });
    h+='</div>';
  });
  return h;
}

function metric(k,v){return '<div class="metric"><div class="k">'+esc(k)+'</div><div class="v">'+v+'</div></div>'}
function stat(n,l){return '<div class="stat"><div class="n">'+n+'</div><div class="l">'+l+'</div></div>'}
function esc(s){if(s==null)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML}

load();
</script>
</body>
</html>"""

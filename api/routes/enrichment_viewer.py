"""
Enrichment Data Viewer
Lightweight UI to browse enrichment data collected per professor per source.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["enrichment"])

PROFILES_DIR = Path(__file__).parent.parent.parent / "output" / "osu_faculty_run" / "profiles"


# ── JSON API routes ──────────────────────────────────────────────────────

@router.get("/api/enrichment/professors")
async def list_professors():
    """Return list of all professors with enrichment status."""
    if not PROFILES_DIR.exists():
        return []

    professors = []
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

        enrichment_json = pdir / "enrichment.json"
        has_enrichment = enrichment_json.exists()
        source_count = 0
        confidence = 0.0
        if has_enrichment:
            try:
                enr = json.loads(enrichment_json.read_text(encoding="utf-8"))
                source_count = enr.get("summary", {}).get("successful_sources", 0)
                confidence = enr.get("confidence", {}).get("overall_confidence", 0.0)
            except Exception:
                pass

        professors.append({
            "profile_id": pdir.name,
            "name": name,
            "has_enrichment": has_enrichment,
            "source_count": source_count,
            "confidence": round(confidence, 3),
        })

    # Sort: enriched first (by source count desc), then alphabetical
    professors.sort(key=lambda p: (-p["source_count"], p["name"].lower()))
    return professors


@router.get("/api/enrichment/professor/{profile_id}")
async def get_professor_enrichment(profile_id: str):
    """Return full enrichment data for a single professor."""
    pdir = PROFILES_DIR / profile_id
    if not pdir.is_dir():
        raise HTTPException(status_code=404, detail="Profile not found")

    # Basic profile info
    profile_json = pdir / f"{profile_id}.json"
    if not profile_json.exists():
        raise HTTPException(status_code=404, detail="Profile JSON not found")

    try:
        profile_data = json.loads(profile_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading profile: {e}")

    result = {
        "profile_id": profile_id,
        "name": profile_data.get("name", ""),
        "profile_url": profile_data.get("profile_url", ""),
        "has_enrichment": False,
        "enrichment": None,
    }

    enrichment_json = pdir / "enrichment.json"
    if enrichment_json.exists():
        try:
            enr = json.loads(enrichment_json.read_text(encoding="utf-8"))
            result["has_enrichment"] = True
            result["enrichment"] = enr
        except Exception:
            pass

    return result


# ── HTML UI route ────────────────────────────────────────────────────────

@router.get("/enrichment", response_class=HTMLResponse)
async def enrichment_viewer_ui():
    """Serve the enrichment data viewer."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enrichment Data Viewer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    min-height: 100vh; padding: 20px; color: #e0e0e0;
}
.container {
    max-width: 1200px; margin: 0 auto;
    background: #1a1a2e; border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    overflow: hidden;
}
.header {
    background: linear-gradient(135deg, #16213e, #0f3460);
    padding: 24px 30px; text-align: center;
}
.header h1 { font-size: 2em; color: #e94560; margin-bottom: 6px; }
.header p { color: #a0a0b0; font-size: 1em; }
.controls {
    padding: 20px 30px; display: flex; gap: 12px; flex-wrap: wrap;
    align-items: center; border-bottom: 1px solid #2a2a3e;
}
.controls select, .controls input {
    padding: 10px 14px; border: 1px solid #3a3a5e;
    border-radius: 8px; font-size: 15px;
    background: #16213e; color: #e0e0e0;
    transition: border-color 0.3s;
}
.controls select:focus, .controls input:focus {
    outline: none; border-color: #e94560;
}
.controls select { flex: 1; min-width: 280px; }
.controls input { flex: 0.5; min-width: 200px; }
.badge-enriched {
    display: inline-block; background: #0f9b58; color: #fff;
    padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-left: 8px;
}
.badge-none {
    display: inline-block; background: #555; color: #aaa;
    padding: 3px 10px; border-radius: 12px; font-size: 0.8em; margin-left: 8px;
}
.content { padding: 20px 30px; }
.prof-info {
    background: #16213e; border-radius: 10px; padding: 20px;
    margin-bottom: 20px; border-left: 4px solid #e94560;
}
.prof-info h2 { color: #e94560; margin-bottom: 8px; }
.prof-info .meta { color: #a0a0b0; font-size: 0.95em; margin: 3px 0; }
.prof-info .meta a { color: #53a8e2; text-decoration: none; }
.prof-info .meta a:hover { text-decoration: underline; }
.summary-bar {
    display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;
}
.stat-card {
    background: #16213e; border-radius: 8px; padding: 14px 20px;
    flex: 1; min-width: 140px; text-align: center;
    border: 1px solid #2a2a3e;
}
.stat-card .num { font-size: 1.8em; font-weight: 700; color: #e94560; }
.stat-card .label { color: #a0a0b0; font-size: 0.85em; margin-top: 2px; }
.sources-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
}
.source-card {
    background: #16213e; border: 1px solid #2a2a3e;
    border-radius: 10px; overflow: hidden;
    transition: border-color 0.3s;
}
.source-card:hover { border-color: #e94560; }
.source-header {
    padding: 14px 18px; display: flex; justify-content: space-between;
    align-items: center; border-bottom: 1px solid #2a2a3e;
    cursor: pointer;
}
.source-header h3 { font-size: 1.05em; color: #e0e0e0; }
.source-status-ok { color: #0f9b58; font-weight: 600; }
.source-status-fail { color: #e94560; font-weight: 600; }
.source-status-cached { color: #f9a825; font-size: 0.8em; margin-left: 4px; }
.source-body {
    padding: 14px 18px; max-height: 400px; overflow-y: auto;
    display: none; font-size: 0.92em; line-height: 1.5;
}
.source-body.open { display: block; }
.source-body pre {
    background: #0d1117; color: #c9d1d9; padding: 12px;
    border-radius: 6px; overflow-x: auto; font-size: 0.85em;
    white-space: pre-wrap; word-break: break-word;
}
.empty-state {
    text-align: center; padding: 60px 20px; color: #666;
}
.empty-state h3 { color: #888; margin-bottom: 8px; }
.loading { text-align: center; padding: 40px; color: #888; }
.confidence-bar {
    height: 6px; border-radius: 3px; background: #2a2a3e;
    overflow: hidden; margin-top: 6px;
}
.confidence-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.5s;
}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Enrichment Data Viewer</h1>
        <p>Browse enrichment data collected from 13 public sources per professor</p>
    </div>
    <div class="controls">
        <select id="professorSelect">
            <option value="">-- Select a professor --</option>
        </select>
        <input type="text" id="searchInput" placeholder="Filter professors...">
    </div>
    <div class="content" id="mainContent">
        <div class="empty-state">
            <h3>Select a professor from the dropdown above</h3>
            <p>Enrichment data from Semantic Scholar, OpenAlex, ORCID, NSF, NIH, and more will be displayed here.</p>
        </div>
    </div>
</div>
<script>
let allProfessors = [];
const ALL_SOURCES = [
    'orcid', 'osu_expertise', 'semantic_scholar', 'openalex', 'crossref',
    'google_scholar', 'nsf_grants', 'nih_grants', 'osu_courses',
    'rate_my_professor', 'osu_news', 'google_news', 'youtube_lectures'
];
const SOURCE_LABELS = {
    'semantic_scholar': 'Semantic Scholar',
    'openalex': 'OpenAlex',
    'google_scholar': 'Google Scholar',
    'crossref': 'CrossRef',
    'orcid': 'ORCID',
    'nsf_grants': 'NSF Grants',
    'nih_grants': 'NIH Grants',
    'rate_my_professor': 'RateMyProfessors',
    'osu_courses': 'OSU Courses',
    'osu_news': 'OSU News',
    'google_news': 'Google News',
    'youtube_lectures': 'YouTube Lectures',
    'osu_expertise': 'OSU Expertise'
};

async function loadProfessors() {
    try {
        const resp = await fetch('/api/enrichment/professors');
        allProfessors = await resp.json();
        populateDropdown(allProfessors);
    } catch(e) {
        document.getElementById('mainContent').innerHTML =
            '<div class="empty-state"><h3>Error loading professors</h3><p>'+esc(e.message)+'</p></div>';
    }
}

function populateDropdown(list) {
    const sel = document.getElementById('professorSelect');
    // Keep the first placeholder option
    sel.innerHTML = '<option value="">-- Select a professor ('+list.length+' total) --</option>';
    list.forEach(p => {
        const badge = p.has_enrichment ? ' ['+p.source_count+' sources]' : '';
        const opt = document.createElement('option');
        opt.value = p.profile_id;
        opt.textContent = p.name + badge;
        sel.appendChild(opt);
    });
}

document.getElementById('searchInput').addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    const filtered = allProfessors.filter(p => p.name.toLowerCase().includes(q));
    populateDropdown(filtered);
});

document.getElementById('professorSelect').addEventListener('change', async e => {
    const pid = e.target.value;
    if (!pid) return;
    const content = document.getElementById('mainContent');
    content.innerHTML = '<div class="loading">Loading enrichment data...</div>';
    try {
        const resp = await fetch('/api/enrichment/professor/'+pid);
        const data = await resp.json();
        renderProfessor(data);
    } catch(e) {
        content.innerHTML = '<div class="empty-state"><h3>Error</h3><p>'+esc(e.message)+'</p></div>';
    }
});

function renderProfessor(data) {
    const content = document.getElementById('mainContent');
    let html = '';

    // Professor info card
    html += '<div class="prof-info">';
    html += '<h2>'+esc(data.name)+'</h2>';
    if (data.profile_url) {
        html += '<div class="meta"><a href="'+esc(data.profile_url)+'" target="_blank">'+esc(data.profile_url)+'</a></div>';
    }
    html += '<div class="meta">Profile ID: '+esc(data.profile_id)+'</div>';
    html += '</div>';

    if (!data.has_enrichment || !data.enrichment) {
        html += '<div class="empty-state">';
        html += '<h3>No enrichment data collected yet</h3>';
        html += '<p>Run the enrichment pipeline for this professor to collect data from public sources.</p>';
        html += '<p style="margin-top:10px;color:#666;font-family:monospace">python run_enrichment_pipeline.py --name "'+esc(data.name)+'"</p>';
        html += '</div>';
        content.innerHTML = html;
        return;
    }

    const enr = data.enrichment;
    const summary = enr.summary || {};
    const confidence = enr.confidence || {};
    const confScore = confidence.overall_confidence || 0;

    // Summary stats bar
    html += '<div class="summary-bar">';
    html += statCard(summary.successful_sources || 0, 'Sources OK');
    html += statCard(summary.failed_sources || 0, 'Failed');
    html += statCard((confScore * 100).toFixed(0) + '%', 'Confidence');
    html += statCard(confidence.name_match_sources || 0, 'Name Matches');
    html += statCard(confidence.osu_affiliation_confirmed || 0, 'OSU Confirmed');
    html += '</div>';

    // Confidence bar
    const confColor = confScore > 0.7 ? '#0f9b58' : confScore > 0.4 ? '#f9a825' : '#e94560';
    html += '<div class="confidence-bar"><div class="confidence-fill" style="width:'+(confScore*100)+'%;background:'+confColor+'"></div></div>';
    html += '<div style="text-align:right;font-size:0.8em;color:#666;margin-bottom:20px">Enriched: '+(enr.enriched_at||'N/A')+'</div>';

    // Sources grid — show ALL 13 sources, successful or not
    html += '<div class="sources-grid">';
    ALL_SOURCES.forEach(srcName => {
        const srcData = (enr.sources || {})[srcName];
        html += renderSourceCard(srcName, srcData);
    });
    html += '</div>';

    content.innerHTML = html;

    // Attach toggle listeners
    document.querySelectorAll('.source-header').forEach(header => {
        header.addEventListener('click', () => {
            const body = header.nextElementSibling;
            body.classList.toggle('open');
        });
    });
}

function renderSourceCard(name, srcData) {
    const label = SOURCE_LABELS[name] || name;
    let html = '<div class="source-card">';
    html += '<div class="source-header">';
    html += '<h3>'+esc(label)+'</h3>';

    if (!srcData) {
        html += '<span class="source-status-fail">Not queried</span>';
        html += '</div>';
        html += '<div class="source-body"><p style="color:#666">This source was not queried during enrichment.</p></div>';
        html += '</div>';
        return html;
    }

    if (srcData.success) {
        html += '<span class="source-status-ok">OK</span>';
        if (srcData.cached) html += '<span class="source-status-cached">cached</span>';
    } else {
        html += '<span class="source-status-fail">Failed</span>';
    }
    html += '</div>';

    // Body
    html += '<div class="source-body">';
    if (srcData.success && srcData.data) {
        const dataObj = srcData.data;
        if (Object.keys(dataObj).length === 0) {
            html += '<p style="color:#888">Source returned no data.</p>';
        } else {
            html += '<pre>'+esc(JSON.stringify(dataObj, null, 2))+'</pre>';
        }
    } else if (srcData.error) {
        html += '<p style="color:#e94560">Error: '+esc(srcData.error)+'</p>';
    } else {
        html += '<p style="color:#888">No data available.</p>';
    }
    html += '</div></div>';
    return html;
}

function statCard(value, label) {
    return '<div class="stat-card"><div class="num">'+value+'</div><div class="label">'+label+'</div></div>';
}

function esc(s) {
    if (s === null || s === undefined) return '';
    const d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

loadProfessors();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

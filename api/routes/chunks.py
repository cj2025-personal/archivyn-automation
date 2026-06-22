"""
Chunks viewer - serve chunk data and simple UI for visualizing extracted_content.json
"""
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import HTMLResponse
from typing import List, Dict
import json
from pathlib import Path
import os

router = APIRouter(prefix="/api", tags=["chunks"])


def _load_stage_cache_profiles() -> Dict[str, Dict]:
    """Load profiles from stage_cache to get CV status"""
    project_root = Path(__file__).parent.parent.parent
    possible_locations = [
        project_root / "stage_cache",
        Path("E:/stage_cache"),
        project_root / "rag_output" / "stage_cache",
        project_root / "output" / "stage_cache",
        Path("rag_output/stage_cache"),
        Path("output/stage_cache"),
    ]
    
    stage_cache_dir = None
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            stage_cache_dir = loc
            break
    
    if stage_cache_dir is None:
        return {}
    
    cache_profiles: Dict[str, Dict] = {}
    json_files = list(stage_cache_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            filename = json_file.name
            parts = json_file.stem.split("_", 1)
            if len(parts) < 2:
                continue
            
            stage_name = parts[0]
            profile_id = parts[1]
            
            if not profile_id:
                continue
            
            with open(json_file, "r", encoding="utf-8") as f:
                stage_data = json.load(f)
            
            # Keep the latest stage for each profile
            if profile_id not in cache_profiles or stage_name > cache_profiles[profile_id]["stage"]:
                cache_profiles[profile_id] = {
                    "stage": stage_name,
                    "data": stage_data,
                }
        except Exception:
            continue
    
    return cache_profiles


def _load_chunked_profiles() -> List[Dict]:
    """Load chunked profiles from output/chunked_profiles"""
    project_root = Path(__file__).parent.parent.parent
    chunked_dir = project_root / "output" / "chunked_profiles"
    
    if not chunked_dir.exists():
        return []
    
    # Load stage_cache profiles to get CV status
    cache_profiles = _load_stage_cache_profiles()
    
    chunked_profiles = []
    profile_dirs = [d for d in chunked_dir.iterdir() if d.is_dir()]
    
    for profile_dir in profile_dirs:
        profile_id = profile_dir.name
        chunks_file = profile_dir / "chunks.json"
        
        if not chunks_file.exists():
            continue
        
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            # Get profile metadata from stage_cache
            profile_meta = {}
            has_cv = False
            if profile_id in cache_profiles:
                cache_data = cache_profiles[profile_id]["data"]
                profile_meta = cache_data.get("profile", {})
                has_cv = bool(profile_meta.get("has_cv", False))
            
            # Convert sections structure to flat chunks list for compatibility
            sections = chunks_data.get("sections", {})
            all_chunks = []
            for section_name, section_chunks in sections.items():
                if isinstance(section_chunks, list):
                    all_chunks.extend(section_chunks)
            
            # Sort by order
            all_chunks.sort(key=lambda x: x.get("order", 0))
            
            chunked_profiles.append({
                "id": profile_id,
                "name": profile_meta.get("name") or profile_meta.get("email") or profile_id,
                "university": profile_meta.get("university", ""),
                "profile_url": profile_meta.get("source", ""),
                "has_cv": has_cv,
                "has_personal_site": bool(profile_meta.get("has_personal_site", False)),
                "chunks": all_chunks,
                "sections": sections,  # Keep sections structure for UI
                "embedding_status": "chunked",
                "source": "chunked_profiles",
                "full_json": chunks_data
            })
        except Exception as e:
            print(f"[ERROR] Failed to load chunked profile {profile_id}: {e}")
            continue
    
    return chunked_profiles


@router.get("/chunks-data")
async def get_chunks_data() -> Dict:
    """
    Return profile summaries and chunks from output/chunked_profiles (prioritized).
    Only includes stage_cache profiles if they don't have chunked versions.
    
    Analytics:
    - total_profiles: total profiles found
    - with_cv: profiles where has_cv == True
    - without_cv: profiles where has_cv == False
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        
        # Load chunked profiles first (these are the priority)
        chunked_profiles = _load_chunked_profiles()
        chunked_profile_ids = {p["id"] for p in chunked_profiles}
        
        # Try to load stage_cache profiles for metadata (CV status, name, etc.)
        # But we'll only use them if we don't have a chunked version
        possible_locations = [
            project_root / "stage_cache",
            Path("E:/stage_cache"),
            project_root / "rag_output" / "stage_cache",
            project_root / "output" / "stage_cache",
            Path("rag_output/stage_cache"),
            Path("output/stage_cache"),
        ]
        
        stage_cache_dir = None
        for loc in possible_locations:
            if loc.exists() and loc.is_dir():
                stage_cache_dir = loc
                break
        
        profiles_list: List[Dict] = []
        seen_ids = set()
        
        # Add chunked profiles first (these are what we want to show)
        for chunked_profile in chunked_profiles:
            profile_id = chunked_profile["id"]
            seen_ids.add(profile_id)
            profiles_list.append(chunked_profile)
        
        # Only add stage_cache profiles if they don't have chunked versions
        if stage_cache_dir:
            json_files = list(stage_cache_dir.glob("*.json"))
            cache_profiles: Dict[str, Dict] = {}
            
            for json_file in json_files:
                try:
                    filename = json_file.name
                    parts = json_file.stem.split("_", 1)
                    if len(parts) < 2:
                        continue
                    
                    stage_name = parts[0]
                    profile_id = parts[1]
                    
                    if not profile_id or profile_id in chunked_profile_ids:
                        continue  # Skip if already have chunked version
                    
                    with open(json_file, "r", encoding="utf-8") as f:
                        stage_data = json.load(f)
                    
                    if profile_id not in cache_profiles or stage_name > cache_profiles[profile_id]["stage"]:
                        cache_profiles[profile_id] = {
                            "stage": stage_name,
                            "data": stage_data,
                            "file": json_file
                        }
                except Exception:
                    continue
            
            # Add non-chunked cache profiles
            for profile_id, cache_info in cache_profiles.items():
                if profile_id in seen_ids:
                    continue
                seen_ids.add(profile_id)
                
                stage_data = cache_info["data"]
                profile_meta = stage_data.get("profile", {})
                
                has_cv = bool(profile_meta.get("has_cv", False))
                chunks = []
                if "chunks" in stage_data:
                    chunks = stage_data["chunks"]

                profile_entry = {
                    "id": profile_id,
                    "name": profile_meta.get("name") or profile_meta.get("email") or profile_id,
                    "university": profile_meta.get("university", ""),
                    "profile_url": profile_meta.get("source", ""),
                    "has_cv": has_cv,
                    "has_personal_site": bool(profile_meta.get("has_personal_site", False)),
                    "chunks": chunks,
                    "embedding_status": f"in_progress_{cache_info['stage']}",
                    "source": "stage_cache",
                    "full_json": stage_data
                }
                profiles_list.append(profile_entry)
        
        # Calculate stats
        total_profiles = len(profiles_list)
        with_cv = sum(1 for p in profiles_list if p.get("has_cv"))
        without_cv = total_profiles - with_cv
        chunked_count = sum(1 for p in profiles_list if p.get("source") == "chunked_profiles")

        return {
            "profiles": profiles_list,
            "stats": {
                "total_profiles": total_profiles,
                "with_cv": with_cv,
                "without_cv": without_cv,
                "from_cache": total_profiles - chunked_count,
                "chunked": chunked_count,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chunks: {e}")


@router.post("/chunks/update-name")
async def update_profile_name(payload: Dict = Body(...)):
    """Update profile_name for a given profile id."""
    profile_id = payload.get("id")
    new_name = payload.get("name")
    if not profile_id or not new_name:
        raise HTTPException(status_code=400, detail="Missing id or name")
    try:
        from api.services.json_writer import get_json_writer
        writer = get_json_writer()
        if not writer.update_profile_name(profile_id, new_name):
            raise HTTPException(status_code=404, detail="Profile not found")
        writer.save()
        return {"status": "ok", "id": profile_id, "name": new_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update name: {e}")


@router.delete("/chunks/profile/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a profile by id from all locations (chunked_profiles, stage_cache, extracted_content.json)."""
    import shutil
    
    try:
        project_root = Path(__file__).parent.parent.parent
        deleted_locations = []
        
        # 1. Delete from chunked_profiles directory
        chunked_dir = project_root / "output" / "chunked_profiles" / profile_id
        if chunked_dir.exists() and chunked_dir.is_dir():
            try:
                shutil.rmtree(chunked_dir)
                deleted_locations.append("chunked_profiles")
                print(f"[Delete] Deleted chunked profile directory: {chunked_dir}")
            except Exception as e:
                print(f"[Delete] Warning: Failed to delete chunked profile directory: {e}")
        
        # 2. Delete from stage_cache (all files matching pattern)
        possible_cache_locations = [
            project_root / "stage_cache",
            Path("E:/stage_cache"),
            project_root / "rag_output" / "stage_cache",
            project_root / "output" / "stage_cache",
            Path("rag_output/stage_cache"),
            Path("output/stage_cache"),
        ]
        
        for cache_dir in possible_cache_locations:
            if cache_dir.exists() and cache_dir.is_dir():
                cache_files = list(cache_dir.glob(f"*_{profile_id}.json"))
                for cache_file in cache_files:
                    try:
                        cache_file.unlink()
                        deleted_locations.append(f"stage_cache/{cache_file.name}")
                        print(f"[Delete] Deleted stage_cache file: {cache_file}")
                    except Exception as e:
                        print(f"[Delete] Warning: Failed to delete stage_cache file {cache_file}: {e}")
        
        # 3. Delete from output/profiles directory
        profile_dir = project_root / "output" / "profiles" / profile_id
        if profile_dir.exists() and profile_dir.is_dir():
            try:
                shutil.rmtree(profile_dir)
                deleted_locations.append("output/profiles")
                print(f"[Delete] Deleted profile directory: {profile_dir}")
            except Exception as e:
                print(f"[Delete] Warning: Failed to delete profile directory: {e}")
        
        # 4. Delete from extracted_content.json (if exists)
        from api.services.json_writer import get_json_writer
        writer = get_json_writer()
        if writer.delete_profile(profile_id):
            writer.save()
            deleted_locations.append("extracted_content.json")
            print(f"[Delete] Deleted from extracted_content.json")
        
        if not deleted_locations:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found in any location")
        
        return {
            "status": "ok", 
            "id": profile_id,
            "deleted_from": deleted_locations
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete profile: {e}")


@router.get("/chunks-viewer", response_class=HTMLResponse)
async def chunks_viewer():
    """
    Simple UI to view chunks by professor via dropdown.
    """
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chunks Viewer</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fb; padding: 30px; }
    .panel { max-width: 1100px; margin: 0 auto; background: #fff; border-radius: 10px; box-shadow: 0 12px 40px rgba(0,0,0,0.08); padding: 24px; }
    h1 { margin: 0 0 12px; color: #2d2f36; }
    p { color: #666; margin: 0 0 16px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; border-bottom: 2px solid #e2e6f0; }
    .tab { padding: 10px 20px; background: transparent; border: none; border-bottom: 3px solid transparent; cursor: pointer; font-size: 14px; font-weight: 500; color: #666; transition: all 0.2s; }
    .tab:hover { color: #667eea; background: #f8f9fc; }
    .tab.active { color: #667eea; border-bottom-color: #667eea; background: #f8f9fc; }
    select { width: 100%; padding: 12px; border: 2px solid #e2e6f0; border-radius: 8px; font-size: 15px; margin-bottom: 18px; }
    .summary { margin: 8px 0 14px; color: #444; font-size: 14px; }
    .chunks { display: grid; gap: 12px; }
    .chunk { background: #f8f9fc; border: 1px solid #e7ebf4; border-radius: 8px; padding: 12px 14px; line-height: 1.5; color: #2d2f36; }
    .meta { font-size: 12px; color: #888; margin-bottom: 6px; }
    .json-section { margin-top: 20px; }
    .json-toggle { background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9em; margin-bottom: 10px; }
    .json-toggle:hover { background: #5568d3; }
    .json-viewer { background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 0.85em; line-height: 1.4; display: none; max-height: 500px; overflow-y: auto; }
    .json-viewer.show { display: block; }
    .json-viewer pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
  </style>
</head>
<body>
  <div class="panel">
    <h1>Chunks Viewer</h1>
    <p>
      View all profiles and their chunks from <code>rag_output/stage_cache</code> (intermediate processing stages).
    </p>
    <div id="analytics" class="summary" style="margin-top:6px; margin-bottom:10px; font-weight:500;"></div>
    <div class="tabs">
      <button class="tab active" id="tabWithCV" onclick="switchTab('with_cv')">Profiles with CV</button>
      <button class="tab" id="tabWithoutCV" onclick="switchTab('without_cv')">Profiles without CV</button>
    </div>
    <div style="display:flex; gap:12px; align-items:center; margin: 0 0 12px;">
      <select id="profSelect" style="flex:1;"></select>
      <label style="font-size:13px; color:#444;">Min chunks
        <input id="minChunks" type="number" min="0" value="0" style="width:80px; margin-left:6px; padding:6px 8px; border:1px solid #dcdfe6; border-radius:6px;">
      </label>
    </div>
    <div style="display:flex; gap:8px; margin: 0 0 12px;">
      <button id="editBtn" style="padding:8px 12px; border:1px solid #dcdfe6; border-radius:6px; background:#eef2fb; cursor:pointer;">Edit name</button>
      <button id="deleteBtn" style="padding:8px 12px; border:1px solid #f1c0c0; border-radius:6px; background:#fdecec; color:#b52020; cursor:pointer;">Delete profile</button>
      <div id="actionStatus" style="font-size:13px; color:#666; padding-left:8px;"></div>
    </div>
    <div id="summary" class="summary"></div>
    <div id="chunks" class="chunks"></div>
    <div class="json-section">
      <button class="json-toggle" onclick="toggleJson()">Show/Hide Raw JSON</button>
      <div class="json-viewer" id="jsonViewer">
        <pre id="jsonContent"></pre>
      </div>
    </div>
  </div>
  <script>
    let profiles = [];
    let filtered = [];
    let lastIndex = 0;
    let activeTab = 'with_cv'; // 'with_cv' or 'without_cv'
    let stats = { total_profiles: 0, with_cv: 0, without_cv: 0 };
    const statusEl = document.getElementById('actionStatus');

    function setStatus(msg, isError=false) {
      statusEl.textContent = msg || '';
      statusEl.style.color = isError ? '#b52020' : '#666';
    }

    async function loadProfiles() {
      setStatus('');
      const res = await fetch('/api/chunks-data');
      const data = await res.json();
      profiles = data.profiles || [];
      stats = data.stats || { total_profiles: 0, with_cv: 0, without_cv: 0 };
      const analyticsEl = document.getElementById('analytics');
      const cacheInfo = stats.from_cache ? ' • In cache: ' + stats.from_cache : '';
      const tabLabel = activeTab === 'with_cv' ? 'profiles with CV' : 'profiles without CV';
      analyticsEl.textContent =
        'Total profiles: ' + (stats.total_profiles || 0) +
        ' • With CV: ' + (stats.with_cv || 0) +
        ' • Without CV: ' + (stats.without_cv || 0) +
        cacheInfo +
        ' • Showing: ' + tabLabel;
      applyFilter();
    }
    function switchTab(tab) {
      activeTab = tab;
      // Update tab buttons
      document.getElementById('tabWithCV').classList.toggle('active', tab === 'with_cv');
      document.getElementById('tabWithoutCV').classList.toggle('active', tab === 'without_cv');
      // Reset selection index when switching tabs
      lastIndex = 0;
      applyFilter();
    }

    function applyFilter() {
      const min = parseInt(document.getElementById('minChunks').value || '0', 10) || 0;
      // First filter by CV status based on active tab
      let tabFiltered = profiles;
      if (activeTab === 'with_cv') {
        tabFiltered = profiles.filter(p => p.has_cv === true);
      } else if (activeTab === 'without_cv') {
        tabFiltered = profiles.filter(p => p.has_cv === false || !p.has_cv);
      }
      // Then filter by min chunks
      filtered = tabFiltered.filter(p => (Array.isArray(p.chunks) ? p.chunks.length : 0) >= min);
      const sel = document.getElementById('profSelect');
      sel.innerHTML = '';
      filtered.forEach((p, idx) => {
        const opt = document.createElement('option');
        const cvBadge = p.has_cv ? ' [CV]' : ' [No CV]';
        const sourceBadge = p.source === 'chunked_profiles' ? ' [Chunked]' : (p.source === 'stage_cache' ? ' [Cache]' : '');
        const chunkCount = p.sections ? 
          Object.values(p.sections).reduce((sum, chunks) => sum + (Array.isArray(chunks) ? chunks.length : 0), 0) :
          (p.chunks ? p.chunks.length : 0);
        opt.value = idx;
        opt.textContent = (p.name || 'Unknown') + cvBadge + sourceBadge + ' (' + chunkCount + ' chunks)';
        sel.appendChild(opt);
      });
      if (filtered.length) {
        sel.value = Math.min(lastIndex, filtered.length - 1);
        renderChunks(sel.value);
      } else {
        document.getElementById('summary').textContent = 'No profiles match this filter.';
        document.getElementById('chunks').innerHTML = '';
      }
    }
    let currentProfileJson = null;
    
    function renderChunks(index) {
      const p = filtered[index];
      lastIndex = Number(index) || 0;
      currentProfileJson = p;
      const container = document.getElementById('chunks');
      const summary = document.getElementById('summary');
      container.innerHTML = '';
      summary.textContent = '';
      
      // Check if we have sections structure (from chunked_profiles)
      if (p?.sections && typeof p.sections === 'object') {
        const sections = p.sections;
        const sectionNames = Object.keys(sections);
        let totalChunks = 0;
        
        sectionNames.forEach(sectionName => {
          const sectionChunks = Array.isArray(sections[sectionName]) ? sections[sectionName] : [];
          totalChunks += sectionChunks.length;
          
          if (sectionChunks.length > 0) {
            // Create section header
            const sectionDiv = document.createElement('div');
            sectionDiv.style.marginBottom = '24px';
            sectionDiv.innerHTML = '<h3 style="margin:0 0 12px; color:#667eea; font-size:18px; border-bottom:2px solid #e2e6f0; padding-bottom:8px;">' + 
                                   escapeHtml(sectionName) + 
                                   ' <span style="font-size:14px; color:#888; font-weight:normal;">(' + sectionChunks.length + ' chunks)</span></h3>';
            
            // Add chunks for this section
            sectionChunks.forEach((c, i) => {
              const chunkDiv = document.createElement('div');
              chunkDiv.className = 'chunk';
              chunkDiv.style.marginBottom = '10px';
              chunkDiv.innerHTML = '<div class="meta">Chunk ' + (i+1) + 
                                   (c.chunk_id ? ' • ID: ' + c.chunk_id.substring(0, 8) + '...' : '') +
                                   (c.order !== undefined ? ' • Order: ' + c.order : '') + '</div>' +
                                   '<div>' + escapeHtml(c.text || '') + '</div>';
              sectionDiv.appendChild(chunkDiv);
            });
            
            container.appendChild(sectionDiv);
          }
        });
        
        summary.textContent = 'Total chunks: ' + totalChunks + ' across ' + sectionNames.length + ' sections';
      } else {
        // Fallback to flat chunks list (for stage_cache profiles)
        const chunksArr = Array.isArray(p?.chunks) ? p.chunks : [];
        summary.textContent = 'Total chunks: ' + (chunksArr.length || 0);
        if (!p || !chunksArr.length) {
          container.innerHTML = '<div class="chunk">No chunks for this profile.</div>';
          return;
        }
        chunksArr.forEach((c, i) => {
          const div = document.createElement('div');
          div.className = 'chunk';
          div.innerHTML = '<div class="meta">Chunk ' + (i+1) + (c.section ? ' • ' + escapeHtml(c.section || '') : '') + '</div>' +
                          '<div>' + escapeHtml(c.text || '') + '</div>';
          container.appendChild(div);
        });
      }
    }
    
    function toggleJson() {
      const viewer = document.getElementById('jsonViewer');
      const content = document.getElementById('jsonContent');
      if (viewer.classList.contains('show')) {
        viewer.classList.remove('show');
      } else {
        viewer.classList.add('show');
        if (currentProfileJson && currentProfileJson.full_json) {
          content.textContent = JSON.stringify(currentProfileJson.full_json, null, 2);
        } else if (currentProfileJson) {
          content.textContent = JSON.stringify(currentProfileJson, null, 2);
        } else {
          content.textContent = 'No profile selected.';
        }
      }
    }
    function escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML.replace(/\\n/g,'<br>');
    }

    async function renameSelected() {
      if (!filtered.length) return;
      const idx = document.getElementById('profSelect').value;
      const prof = filtered[idx];
      const newName = prompt('Enter new name for profile', prof.name || '');
      if (!newName || newName.trim() === '' || newName === prof.name) return;
      try {
        setStatus('Saving...');
        const resp = await fetch('/api/chunks/update-name', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ id: prof.id, name: newName.trim() })
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to update');
        await loadProfiles();
        setStatus('Name updated.');
      } catch (e) {
        setStatus(e.message || 'Update failed', true);
      }
    }

    async function deleteSelected() {
      if (!filtered.length) return;
      const idx = document.getElementById('profSelect').value;
      const prof = filtered[idx];
      if (!confirm(`Delete profile \"${prof.name}\"?`)) return;
      try {
        setStatus('Deleting...');
        const resp = await fetch(`/api/chunks/profile/${encodeURIComponent(prof.id)}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to delete');
        await loadProfiles();
        setStatus('Profile deleted.');
      } catch (e) {
        setStatus(e.message || 'Delete failed', true);
      }
    }

    document.getElementById('profSelect').addEventListener('change', (e)=> renderChunks(e.target.value));
    document.getElementById('minChunks').addEventListener('change', applyFilter);
    document.getElementById('editBtn').addEventListener('click', renameSelected);
    document.getElementById('deleteBtn').addEventListener('click', deleteSelected);
    loadProfiles();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

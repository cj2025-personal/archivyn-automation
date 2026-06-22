"""
Vector DB Viewer - Interface to view and search records from Pinecone
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

router = APIRouter(prefix="/api", tags=["view-db"])


class ProfessorRecord(BaseModel):
    id: str
    professor_name: str
    university: str
    department: str
    email: str
    position: str
    profile_url: str
    content: str  # Full stored content (aggregated from chunks if chunked)
    content_preview: str
    content_length: int
    content_truncated: Optional[bool] = False
    total_chunks: Optional[int] = 1  # Number of chunks this content was stored in
    score: float
    metadata: Dict


class ViewDBResponse(BaseModel):
    total: int
    records: List[ProfessorRecord]
    stats: Dict


@router.get("/view-db", response_model=ViewDBResponse)
async def view_db(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    university: Optional[str] = Query(None, description="Filter by university"),
    department: Optional[str] = Query(None, description="Filter by department"),
    search: Optional[str] = Query(None, description="Text search query")
):
    """
    Get all professor records from the vector database
    
    Args:
        limit: Maximum number of records to return (1-1000)
        university: Optional filter by university
        department: Optional filter by department
        search: Optional text search query
    
    Returns:
        List of professor records with metadata
    """
    try:
        from api.services.vector_db import get_vector_db
        from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
        
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        
        # Get stats
        stats = vector_db.get_stats()
        
        # Get records
        if search or university or department:
            records = vector_db.search_professors(
                query_text=search or "",
                university=university or "",
                department=department or "",
                limit=limit
            )
        else:
            records = vector_db.get_all_professors(limit=limit)
        
        return ViewDBResponse(
            total=len(records),
            records=records,
            stats=stats
        )
        
    except Exception as e:
        import traceback
        error_msg = str(e) if str(e) else repr(e)
        print(f"[ViewDB] Error: {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve records: {error_msg}")


@router.get("/view-db-interface", response_class=HTMLResponse)
async def view_db_interface():
    """
    Serve the vector DB viewer web interface
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vector DB Viewer - Professor Records</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .filters {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            align-items: end;
        }
        
        .filter-group {
            display: flex;
            flex-direction: column;
        }
        
        .filter-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 0.9em;
        }
        
        .filter-group input,
        .filter-group select {
            padding: 10px 15px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .filter-group input:focus,
        .filter-group select:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .search-button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .search-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
        }
        
        .stats-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #e8f4f8;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .records-container {
            display: grid;
            gap: 20px;
        }
        
        .record-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 25px;
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .record-card:hover {
            border-color: #667eea;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        
        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .record-title {
            flex: 1;
        }
        
        .record-title h3 {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 5px;
        }
        
        .record-title .subtitle {
            color: #666;
            font-size: 0.95em;
        }
        
        .record-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .record-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            font-size: 0.85em;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 1em;
            color: #333;
            font-weight: 500;
        }
        
        .detail-value.empty {
            color: #ccc;
            font-style: italic;
        }
        
        .content-preview {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 0.9em;
            line-height: 1.6;
            color: #555;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .content-preview::-webkit-scrollbar {
            width: 8px;
        }
        
        .content-preview::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .content-preview::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .empty-state-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            margin-bottom: 20px;
        }
        
        .expand-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 10px;
            transition: background 0.3s;
        }
        
        .expand-button:hover {
            background: #5568d3;
        }
        
        .expanded-content {
            display: none;
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        .expanded-content.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Vector DB Viewer</h1>
            <p>View and search professor records from Pinecone</p>
        </div>
        
        <div class="content">
            <div class="filters">
                <div class="filter-group">
                    <label for="searchInput">🔍 Search Query</label>
                    <input type="text" id="searchInput" placeholder="Search by name, research, etc...">
                </div>
                <div class="filter-group">
                    <label for="universityFilter">🏛️ University</label>
                    <input type="text" id="universityFilter" placeholder="Filter by university">
                </div>
                <div class="filter-group">
                    <label for="departmentFilter">📚 Department</label>
                    <input type="text" id="departmentFilter" placeholder="Filter by department">
                </div>
                <div class="filter-group">
                    <label for="limitInput">📊 Limit</label>
                    <input type="number" id="limitInput" value="100" min="1" max="1000">
                </div>
                <div class="filter-group">
                    <button class="search-button" onclick="loadRecords()">🔍 Search</button>
                </div>
            </div>
            
            <div class="stats-bar" id="statsBar" style="display: none;">
                <div class="stat-item">
                    <div class="stat-value" id="totalRecords">0</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="totalVectors">0</div>
                    <div class="stat-label">Total Vectors</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="dimension">0</div>
                    <div class="stat-label">Dimension</div>
                </div>
            </div>
            
            <div id="errorMessage" class="error-message" style="display: none;"></div>
            
            <div id="loading" class="loading" style="display: none;">
                <div class="loading-spinner"></div>
                <p>Loading records...</p>
            </div>
            
            <div id="recordsContainer" class="records-container"></div>
            
            <div id="emptyState" class="empty-state" style="display: none;">
                <div class="empty-state-icon">📭</div>
                <h3>No records found</h3>
                <p>Try adjusting your filters or search query</p>
            </div>
        </div>
    </div>
    
    <script>
        async function loadRecords() {
            const search = document.getElementById('searchInput').value.trim();
            const university = document.getElementById('universityFilter').value.trim();
            const department = document.getElementById('departmentFilter').value.trim();
            const limit = parseInt(document.getElementById('limitInput').value) || 100;
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('recordsContainer').innerHTML = '';
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('statsBar').style.display = 'none';
            
            try {
                // Build query params
                const params = new URLSearchParams();
                params.append('limit', limit.toString());
                if (search) params.append('search', search);
                if (university) params.append('university', university);
                if (department) params.append('department', department);
                
                const response = await fetch(`/api/view-db?${params.toString()}`);
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to load records');
                }
                
                const data = await response.json();
                
                // Update stats
                document.getElementById('totalRecords').textContent = data.total;
                document.getElementById('totalVectors').textContent = data.stats.total_vectors || 0;
                document.getElementById('dimension').textContent = data.stats.dimension || 0;
                document.getElementById('statsBar').style.display = 'flex';
                
                // Display records
                if (data.records && data.records.length > 0) {
                    displayRecords(data.records);
                } else {
                    document.getElementById('emptyState').style.display = 'block';
                }
                
            } catch (error) {
                document.getElementById('errorMessage').textContent = `Error: ${error.message}`;
                document.getElementById('errorMessage').style.display = 'block';
                console.error(error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayRecords(records) {
            const container = document.getElementById('recordsContainer');
            container.innerHTML = '';
            
            records.forEach((record, index) => {
                const card = document.createElement('div');
                card.className = 'record-card';
                card.innerHTML = `
                    <div class="record-header">
                        <div class="record-title">
                            <h3>${escapeHtml(record.professor_name || 'Unknown')}</h3>
                            <div class="subtitle">${escapeHtml(record.university || 'N/A')} ${record.department ? '• ' + escapeHtml(record.department) : ''}</div>
                        </div>
                        <div class="record-score">Score: ${record.score.toFixed(4)}</div>
                    </div>
                    
                    <div class="record-details">
                        <div class="detail-item">
                            <div class="detail-label">Email</div>
                            <div class="detail-value ${!record.email ? 'empty' : ''}">${escapeHtml(record.email || 'Not provided')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Position</div>
                            <div class="detail-value ${!record.position ? 'empty' : ''}">${escapeHtml(record.position || 'Not provided')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Department</div>
                            <div class="detail-value ${!record.department ? 'empty' : ''}">${escapeHtml(record.department || 'Not provided')}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Content Length</div>
                            <div class="detail-value">${record.content_length.toLocaleString()} chars${record.total_chunks > 1 ? ` (${record.total_chunks} chunks)` : ''}</div>
                        </div>
                    </div>
                    
                    ${record.profile_url ? `<div class="detail-item" style="margin-top: 10px;">
                        <div class="detail-label">Profile URL</div>
                        <div class="detail-value"><a href="${escapeHtml(record.profile_url)}" target="_blank" style="color: #667eea; text-decoration: none;">${escapeHtml(record.profile_url)}</a></div>
                    </div>` : ''}
                    
                    <div class="content-preview" id="preview-${index}">
                        ${escapeHtml((record.content || record.content_preview || 'No content available').substring(0, 500))}
                        ${record.content_length > 500 ? '...' : ''}
                    </div>
                    
                    ${record.content && record.content.length > 500 ? `<button class="expand-button" onclick="toggleExpand(${index})">${record.content_truncated ? 'Show Stored Content' : 'Show Full Content'}</button>
                    <div class="expanded-content" id="expanded-${index}"></div>` : ''}
                    ${record.content_truncated ? `<div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; color: #856404;">
                        ⚠️ Content was truncated. Original length: ${record.content_length.toLocaleString()} chars, stored: ${(record.content || '').length.toLocaleString()} chars
                    </div>` : ''}
                `;
                
                container.appendChild(card);
                
                // Store full content for expand functionality
                if (record.content && record.content.length > 500) {
                    card.dataset.fullContent = record.content || '';
                }
            });
        }
        
        function toggleExpand(index) {
            const expanded = document.getElementById(`expanded-${index}`);
            const button = event.target;
            const card = button.closest('.record-card');
            const fullContent = card.dataset.fullContent || '';
            const isTruncated = card.querySelector('.expanded-content').previousElementSibling && 
                                card.querySelector('.expanded-content').previousElementSibling.textContent.includes('truncated');
            
            if (expanded.classList.contains('show')) {
                expanded.classList.remove('show');
                button.textContent = isTruncated ? 'Show Stored Content' : 'Show Full Content';
            } else {
                // Display full stored content
                expanded.textContent = fullContent || 'Full content not available';
                expanded.classList.add('show');
                button.textContent = 'Hide Content';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load records on page load
        window.addEventListener('load', () => {
            loadRecords();
        });
        
        // Allow Enter key to trigger search
        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadRecords();
            }
        });
        
        document.getElementById('universityFilter').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadRecords();
            }
        });
        
        document.getElementById('departmentFilter').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                loadRecords();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


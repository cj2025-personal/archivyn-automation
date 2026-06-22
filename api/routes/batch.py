"""
Batch processing endpoint - processes multiple profile URLs from Excel file
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from api.services.scraper import get_scraper
import pandas as pd
import asyncio
import io

router = APIRouter(prefix="/api", tags=["batch"])


class BatchProcessResponse(BaseModel):
    total_urls: int
    processed: int
    successful: int
    failed: int
    results: List[Dict]
    errors: List[Dict]


@router.post("/batch-process", response_model=BatchProcessResponse)
async def batch_process(file: UploadFile = File(...)):
    """
    Process multiple profile URLs from an uploaded Excel file
    
    Expected Excel format:
    - Preferred column for URL: source (will be treated as profile_url)
    - Fallback column: profile_url (required if source is missing)
    - Optional columns: name, email, university, department, etc.
    
    Returns:
        - total_urls: Total URLs processed in this run
        - processed: Number of URLs processed
        - successful: Number of successful extractions
        - failed: Number of failed extractions
        - results: List of extraction results
        - errors: List of errors encountered
    """
    try:
        # Read Excel file
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="File must be an Excel file (.xlsx or .xls)"
            )
        
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Validate required column - prefer 'source' as URL column, fallback to profile_url
        if 'source' in df.columns and 'profile_url' not in df.columns:
            df.rename(columns={'source': 'profile_url'}, inplace=True)
        elif 'profile_url' not in df.columns:
            # Try common variations
            url_columns = [col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()]
            if url_columns:
                df.rename(columns={url_columns[0]: 'profile_url'}, inplace=True)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Excel file must contain a 'source' or 'profile_url' column"
                )
        
        # Get URLs (remove empty rows)
        valid_rows = df[df['profile_url'].notna()].reset_index(drop=True)
        if valid_rows.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid URLs found in the Excel file"
            )
        
        rows_to_process = valid_rows
        total_urls = len(rows_to_process)
        print(f"[Batch] Processing all rows. Total: {total_urls}")
        
        # Process URLs
        scraper = await get_scraper()
        results = []
        errors = []
        successful = 0
        failed = 0
        
        # Vector DB storage is DISABLED
        # Initialize vector DB and embeddings service
        vector_db = None
        embeddings_service = None
        vector_db_enabled = False
        
        # DISABLED: Vector DB storage logic commented out
        # Uncomment below to re-enable vector storage
        """
        try:
            from api.services.vector_db import get_vector_db
            from api.services.embeddings import get_embeddings_service
            from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
            
            print(f"[Batch] Initializing Vector DB: {INDEX_NAME} (dimension: {INDEX_DIMENSION})")
            vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
            print(f"[Batch] Vector DB connected successfully")
            
            print(f"[Batch] Initializing embeddings service...")
            embeddings_service = get_embeddings_service()
            print(f"[Batch] Embeddings service initialized: {embeddings_service.model_name} (dimension: {embeddings_service.get_dimension()})")
            
            vector_db_enabled = True
            print(f"[Batch] ✅ Vector DB storage ENABLED")
        except Exception as e:
            import traceback
            print(f"[Batch] ❌ ERROR: Could not initialize vector DB: {str(e)}")
            print(f"[Batch] Full error traceback:")
            traceback.print_exc()
            print(f"[Batch] Continuing without vector DB storage...")
            vector_db_enabled = False
        """
        
        print(f"[Batch] Vector DB storage is DISABLED")
        
        for idx, (_, row) in enumerate(rows_to_process.iterrows(), 1):
            try:
                url = row['profile_url']
                current_row_number = idx  # 1-based row number within valid URLs
                print(f"[Batch] Processing {idx}/{total_urls} (row #{current_row_number}): {str(url)[:60]}...")
                result = await scraper.extract_all(str(url))
                
                # Get additional data from Excel row if available
                row_data = {}
                for col in df.columns:
                    if col != 'profile_url' and pd.notna(row.get(col)):
                        row_data[col] = str(row[col])
                
                profile_data = result.get('profile_data', {})
                
                # Save profile content to JSON with chunks using data cleaning pipeline
                try:
                    from api.services.json_writer import get_json_writer
                    from api.services.data_cleaning_service import get_data_cleaning_service
                    import uuid
                    
                    json_writer = get_json_writer()
                    
                    # Get profile information
                    profile_name = profile_data.get('name', '') or row_data.get('name', '') or 'Unknown'
                    profile_id = str(uuid.uuid4())[:8]
                    
                    # Collect all URLs
                    all_urls = [str(url)]  # Start with profile URL
                    body_links = result.get('body_links_with_content', [])
                    for link in body_links:
                        if link.get('source_url'):
                            all_urls.append(link.get('source_url'))
                    
                    # Combine all text content (same as extract.py)
                    combined_text_parts = []
                    combined_headings = []
                    combined_paragraphs = []
                    
                    # Add profile page content
                    profile_text = result.get('text_content', {})
                    if profile_text.get('full_text'):
                        combined_text_parts.append(f"=== PROFILE PAGE ===\n{profile_text.get('full_text', '')}")
                        combined_headings.extend(profile_text.get('headings', []))
                        combined_paragraphs.extend(profile_text.get('paragraphs', []))
                    
                    # Add body links content (documents and webpages)
                    for link in body_links:
                        if link.get('content') and link.get('processing_status') == 'success':
                            source_type = link.get('source_type', 'webpage')
                            source_url = link.get('source_url', '')
                            link_text = link.get('link_text', '') or 'Content'
                            
                            # Add separator with source info
                            combined_text_parts.append(f"\n\n=== {source_type.upper()}: {link_text} ({source_url}) ===\n{link.get('content', '')}")
                    
                    # Combine all text
                    combined_text = '\n\n'.join(combined_text_parts) if combined_text_parts else ''
                    
                    # Debug: Print what we have
                    print(f"[Batch] Debug - combined_text length: {len(combined_text)}")
                    print(f"[Batch] Debug - profile_text.get('full_text') length: {len(profile_text.get('full_text', ''))}")
                    print(f"[Batch] Debug - body_links count: {len(body_links)}")
                    print(f"[Batch] Debug - combined_text_parts count: {len(combined_text_parts)}")
                    
                    # Only process if we have content - be more lenient with the condition
                    chunks = []
                    has_content = bool(combined_text.strip()) or bool(profile_text.get('full_text', '').strip()) or len(body_links) > 0
                    
                    if has_content:
                        print(f"[Batch] Processing profile {idx}/{total_urls}: {profile_name}")
                        print(f"[Batch] URLs: {len(all_urls)}, Text parts: {len(combined_text_parts)}")
                        
                        # Use the data cleaning pipeline for chunking
                        try:
                            # Initialize the data cleaning service with optimal chunk sizes
                            cleaning_service = get_data_cleaning_service(
                                target_words_per_chunk=325,
                                min_words_per_chunk=250,
                                max_words_per_chunk=400
                            )
                            
                            # Process text through the complete pipeline
                            print(f"[Batch] Processing text through data cleaning pipeline...")
                            pipeline_chunks = cleaning_service.clean_and_chunk_text(
                                text=combined_text,
                                profile_url=str(url),
                                section_header=""
                            )
                            
                            # Convert pipeline chunks to the format expected by json_writer
                            for chunk_idx, chunk in enumerate(pipeline_chunks):
                                chunks.append({
                                    'chunk_id': chunk.get('id', f"chunk_{chunk_idx}"),
                                    'chunk_index': chunk_idx,
                                    'text': chunk.get('text', ''),
                                    'char_count': chunk.get('metadata', {}).get('char_count', len(chunk.get('text', ''))),
                                    'word_count': chunk.get('metadata', {}).get('length', len(chunk.get('text', '').split())),
                                    'professor_name': profile_name,
                                    'section': chunk.get('metadata', {}).get('section', ''),
                                    'heading': chunk.get('metadata', {}).get('section', ''),
                                    'section_type': chunk.get('metadata', {}).get('section', '').lower() if chunk.get('metadata', {}).get('section') else 'other',
                                    'source': 'data_cleaning_pipeline',
                                    'start_position': 0,
                                    'end_position': chunk.get('metadata', {}).get('char_count', len(chunk.get('text', '')))
                                })
                            
                            print(f"[Batch] Created {len(chunks)} chunks using data cleaning pipeline")
                            
                        except Exception as pipeline_error:
                            import traceback
                            print(f"[Batch] Warning: Data cleaning pipeline failed: {str(pipeline_error)}")
                            print(traceback.format_exc())
                            print(f"[Batch] Profile will be saved without chunks")
                        
                        # Write profile content to JSON (with chunks)
                        json_writer.write_profile_content(
                            profile_id=profile_id,
                            profile_name=profile_name,
                            profile_url=str(url),
                            all_urls=all_urls,
                            combined_text=combined_text,
                            combined_headings=combined_headings,
                            combined_paragraphs=combined_paragraphs,
                            cleaning_status='cleaned' if chunks else 'pending',
                            chunks=chunks
                        )
                        
                        # Save JSON file after each profile (or could batch save at end)
                        json_writer.save()
                        print(f"[Batch] ✅ Saved profile to JSON - ID: {profile_id}, Name: {profile_name}, Chunks: {len(chunks)}")
                    else:
                        print(f"[Batch] ⚠️ No content to save for profile: {profile_name}")
                        print(f"[Batch] Debug - combined_text empty: {not combined_text.strip()}")
                        print(f"[Batch] Debug - profile_text empty: {not profile_text.get('full_text', '').strip()}")
                        print(f"[Batch] Debug - body_links empty: {len(body_links) == 0}")
                        # Still save the profile even if no content, so we have a record
                        json_writer.write_profile_content(
                            profile_id=profile_id,
                            profile_name=profile_name,
                            profile_url=str(url),
                            all_urls=all_urls,
                            combined_text=combined_text or '',
                            combined_headings=combined_headings or [],
                            combined_paragraphs=combined_paragraphs or [],
                            cleaning_status='no_content',
                            chunks=[]
                        )
                        json_writer.save()
                        print(f"[Batch] Saved profile with no content - ID: {profile_id}, Name: {profile_name}")
                    
                    # Add to results after successful JSON save
                    results.append({
                        'url': str(url),
                        'index': current_row_number,
                        'status': 'success',
                        'profile_data': profile_data,
                        'all_urls': result.get('all_urls', []),
                        'document_links': result.get('document_links', []),
                        'text_content': result.get('text_content', {}),
                        'body_links_with_content': result.get('body_links_with_content', []),
                        'extraction_metadata': result.get('extraction_metadata', {}),
                        'excel_data': row_data,
                        'chunks_count': len(chunks),
                        'profile_id': profile_id,
                        'profile_name': profile_name
                    })
                    successful += 1
                        
                except Exception as json_error:
                    import traceback
                    print(f"[Batch] ❌ ERROR: Failed to save to JSON: {str(json_error)}")
                    print(f"[Batch] Full traceback:")
                    traceback.print_exc()
                    # Still add to results but mark as error
                    errors.append({
                        'url': str(url),
                        'index': current_row_number,
                        'error': f"JSON save failed: {str(json_error)}"
                    })
                    # Don't fail the batch processing if JSON save fails, but log it
                    # Still add to results
                    results.append({
                        'url': str(url),
                        'index': current_row_number,
                        'status': 'error',
                        'error': f"JSON save failed: {str(json_error)}"
                    })
                    failed += 1
                
                # DISABLED: Vector DB storage logic commented out
                # Store in vector DB if available - AGGREGATE ALL CONTENT PER PROFESSOR
                """
                if vector_db_enabled and vector_db and embeddings_service and profile_data:
                    try:
                        print(f"[Batch] Starting vector DB storage for profile {idx}...")
                        
                        # Get professor name (from profile_data or Excel row)
                        professor_name = profile_data.get('name', '') or row_data.get('name', '')
                        if not professor_name:
                            # Try to extract from URL or use a default
                            professor_name = f"Unknown_{idx}"
                            print(f"[Batch] ⚠️ No professor name found, using: {professor_name}")
                        
                        print(f"[Batch] Professor: {professor_name}")
                        
                        # Aggregate all content: profile + documents + webpages (raw, no cleaning)
                        combined_content_parts = []
                        
                        # 1. Add profile data (raw, no cleaning)
                        profile_text = profile_data.get('full_text', '')
                        if profile_text:
                            combined_content_parts.append(f"PROFILE:\n{profile_text}\n")
                        
                        # Add other profile fields (raw, no cleaning)
                        if profile_data.get('bio'):
                            bio = profile_data.get('bio', '')
                            combined_content_parts.append(f"BIO:\n{bio}\n")
                        
                        # 2. Add all document and webpage content (already cleaned by scraper)
                        body_links = result.get('body_links_with_content', [])
                        print(f"[Batch] Found {len(body_links)} links with content")
                        for link_idx, link in enumerate(body_links):
                            if link.get('content') and link.get('processing_status') == 'success':
                                content = link.get('content', '')
                                if content and len(content) > 50:  # Only include substantial content
                                    source_type = link.get('source_type', 'content')
                                    source_url = link.get('source_url', '')
                                    
                                    # Content is raw (no cleaning), just add it
                                    combined_content_parts.append(f"\n{source_type.upper()} ({source_url}):\n{content}\n")
                                    print(f"[Batch] Added {source_type} {link_idx+1}/{len(body_links)}: {len(content)} chars")
                        
                        # Combine all content (raw, no cleaning)
                        combined_content = "\n".join(combined_content_parts)
                        
                        if not combined_content.strip():
                            print(f"[Batch] ⚠️ No content to store for {professor_name}")
                            continue
                        
                        print(f"[Batch] Aggregated content for {professor_name}: {len(combined_content)} characters (raw, no cleaning)")
                        
                        # Generate embedding for combined content
                        print(f"[Batch] Generating embedding for {professor_name}...")
                        print(f"[Batch] Content to embed: {len(combined_content)} characters")
                        if not combined_content or not combined_content.strip():
                            print(f"[Batch] ❌ ERROR: Combined content is empty for {professor_name}")
                            continue
                        
                        # Note: embed_text will automatically truncate if content is too long
                        # OpenAI text-embedding-3-small has 8192 token limit (~28,000 chars)
                        try:
                            combined_embedding = embeddings_service.embed_text(combined_content, max_tokens=8000)
                            print(f"[Batch] ✅ Embedding generated: {len(combined_embedding)} dimensions")
                            
                            # Validate embedding before storing
                            if all(v == 0.0 for v in combined_embedding):
                                print(f"[Batch] ❌ ERROR: Generated embedding is all zeros for {professor_name}")
                                print(f"[Batch] This usually means embedding generation failed")
                                continue
                            
                            non_zero = sum(1 for v in combined_embedding if v != 0.0)
                            print(f"[Batch] Embedding validation: {non_zero}/{len(combined_embedding)} non-zero values")
                            
                        except Exception as emb_error:
                            print(f"[Batch] ❌ ERROR generating embedding: {str(emb_error)}")
                            import traceback
                            traceback.print_exc()
                            # Don't raise - continue to next professor
                            continue
                        
                        # Prepare metadata
                        metadata = {
                            'profile_url': str(url),
                            'university': profile_data.get('university', '') or row_data.get('university', ''),
                            'department': profile_data.get('department', '') or row_data.get('department', ''),
                            'email': profile_data.get('email', '') or row_data.get('email', ''),
                            'position': profile_data.get('position', '') or row_data.get('position', ''),
                        }
                        if row_data:
                            metadata.update({k: v for k, v in row_data.items() if k not in metadata})
                        
                        # Store single vector per professor
                        print(f"[Batch] Storing aggregated content for {professor_name} in vector DB...")
                        try:
                            success = vector_db.upsert_professor(
                                professor_name=professor_name,
                                combined_content=combined_content,
                                embedding=combined_embedding,
                                metadata=metadata
                            )
                            if success:
                                print(f"[Batch] ✅ Successfully stored professor: {professor_name}")
                            else:
                                print(f"[Batch] ⚠️ Professor storage returned False: {professor_name}")
                        except Exception as upsert_error:
                            print(f"[Batch] ❌ ERROR in upsert_professor: {str(upsert_error)}")
                            import traceback
                            traceback.print_exc()
                            raise
                    except Exception as vec_error:
                        import traceback
                        print(f"[Batch] ❌ ERROR storing in vector DB: {str(vec_error)}")
                        print(f"[Batch] Full error traceback:")
                        traceback.print_exc()
                        # Continue processing even if vector DB fails
                elif not vector_db_enabled:
                    print(f"[Batch] ⚠️ Vector DB storage is DISABLED (initialization failed)")
                """
                # Vector DB storage is disabled - no vectors will be stored
                # Results are already appended above after JSON save
                
            except Exception as e:
                error_msg = str(e) if str(e) else repr(e)
                print(f"[Batch] Error processing row #{current_row_number} {str(url)[:60]}...: {error_msg}")
                
                errors.append({
                    'url': str(url),
                    'index': current_row_number,
                    'error': error_msg
                })
                results.append({
                    'url': str(url),
                    'index': current_row_number,
                    'status': 'error',
                    'error': error_msg
                })
                failed += 1
        
        return BatchProcessResponse(
            total_urls=total_urls,
            processed=successful + failed,
            successful=successful,
            failed=failed,
            results=results,
            errors=errors
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        import traceback
        print(f"Batch processing error: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {error_msg}")


@router.get("/batch-interface", response_class=HTMLResponse)
async def batch_interface():
    """
    Serve the batch processing web interface
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Faculty Profile Batch Processor</title>
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
            max-width: 1400px;
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
        
        .upload-section {
            background: #f8f9fa;
            border: 2px dashed #667eea;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f0f0;
        }
        
        .upload-section.dragover {
            border-color: #764ba2;
            background: #e8e8ff;
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            margin: 20px 0;
        }
        
        .file-input {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }
        
        .file-input-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
            display: inline-block;
        }
        
        .file-input-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .file-name {
            margin-top: 15px;
            color: #666;
            font-size: 0.95em;
        }
        
        .process-button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px 50px;
            border: none;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: all 0.3s ease;
            display: none;
        }
        
        .process-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4);
        }
        
        .process-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .progress-section {
            display: none;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .stat-card {
            flex: 1;
            min-width: 150px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        
        .message-section {
            margin-top: 30px;
            display: none;
            text-align: center;
        }
        
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 20px 40px;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
            font-size: 1.2em;
            font-weight: 600;
            display: inline-block;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 20px 40px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            font-size: 1.2em;
            font-weight: 600;
            display: inline-block;
            max-width: 800px;
            word-break: break-word;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .instructions {
        .instructions {
            background: #e8f4f8;
            border-left: 4px solid #11998e;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 30px;
        }
        
        .instructions h3 {
            color: #11998e;
            margin-bottom: 10px;
        }
        
        .instructions ul {
            margin-left: 20px;
            color: #555;
        }
        
        .instructions li {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Faculty Profile Batch Processor</h1>
            <p>Upload an Excel file with profile URLs (preferably in a 'source' column) to extract data in bulk</p>
        </div>
        
        <div class="content">
            <div class="instructions">
                <h3>📋 Instructions</h3>
                <ul>
                    <li>Your Excel file should contain a column named <strong>source</strong> (preferred) or <strong>profile_url</strong></li>
                    <li>Each row should contain a valid profile URL in that column</li>
                    <li>Additional columns (name, email, university, etc.) will be preserved in results</li>
                    <li>Supported formats: .xlsx, .xls</li>
                </ul>
            </div>
            
            <div class="upload-section" id="uploadSection">
                <h2>📁 Upload Excel File</h2>
                <p style="color: #666; margin: 15px 0;">Drag and drop your Excel file here, or click to browse</p>
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" class="file-input" accept=".xlsx,.xls">
                    <label for="fileInput" class="file-input-button">Choose File</label>
                </div>
                <div class="file-name" id="fileName"></div>
                <button class="process-button" id="processButton">🚀 Process URLs</button>
            </div>
            
            <div class="progress-section" id="progressSection">
                <h3>Processing...</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="totalStat">0</div>
                        <div class="stat-label">Total URLs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="processedStat">0</div>
                        <div class="stat-label">Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="successStat">0</div>
                        <div class="stat-label">Successful</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="failedStat">0</div>
                        <div class="stat-label">Failed</div>
                    </div>
                </div>
            </div>
            
            <div class="message-section" id="messageSection">
                <div id="messageContent"></div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        let results = null;
        
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const processButton = document.getElementById('processButton');
        const uploadSection = document.getElementById('uploadSection');
        const progressSection = document.getElementById('progressSection');
        const messageSection = document.getElementById('messageSection');
        const messageContent = document.getElementById('messageContent');
        
        // File input handler
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                fileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                processButton.style.display = 'inline-block';
            }
        });
        
        // Drag and drop handlers
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && (file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
                selectedFile = file;
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                processButton.style.display = 'inline-block';
            } else {
                alert('Please upload a valid Excel file (.xlsx or .xls)');
            }
        });
        
        // Process button handler
        processButton.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('Please select a file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
                // Show progress
                processButton.disabled = true;
                processButton.innerHTML = 'Processing<span class="loading"></span>';
                progressSection.style.display = 'block';
                messageSection.style.display = 'none';
            
            try {
                const response = await fetch('/api/batch-process', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Processing failed');
                }
                
                results = await response.json();
                
                // Update stats
                document.getElementById('totalStat').textContent = results.total_urls;
                document.getElementById('processedStat').textContent = results.processed;
                document.getElementById('successStat').textContent = results.successful;
                document.getElementById('failedStat').textContent = results.failed;
                document.getElementById('progressFill').style.width = '100%';
                document.getElementById('progressFill').textContent = '100%';
                
                // Hide progress, show message
                progressSection.style.display = 'none';
                messageSection.style.display = 'block';
                
                // Show success or error message
                if (results.successful > 0 || results.processed > 0) {
                    messageContent.innerHTML = '<div class="success-message">✅ Uploaded successfully</div>';
                } else {
                    const errorMsg = results.errors && results.errors.length > 0 
                        ? results.errors[0].error 
                        : 'Processing failed';
                    messageContent.innerHTML = `<div class="error-message">❌ Error: ${errorMsg}</div>`;
                }
                
            } catch (error) {
                // Show error message
                progressSection.style.display = 'none';
                messageSection.style.display = 'block';
                messageContent.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
                console.error(error);
            } finally {
                processButton.disabled = false;
                processButton.innerHTML = '🚀 Process URLs';
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


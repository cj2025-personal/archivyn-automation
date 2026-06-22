"""
Extract endpoint - extracts all URLs and documents from profile page
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from api.services.scraper import get_scraper
import asyncio

router = APIRouter(prefix="/api", tags=["extract"])


class ExtractRequest(BaseModel):
    profile_url: str


class ExtractResponse(BaseModel):
    profile_data: Dict
    all_urls: List[Dict]
    document_links: List[Dict]
    text_content: Dict
    body_links_with_content: List[Dict] = []  # Links with raw content extracted
    extraction_metadata: Dict


@router.post("/extract-all", response_model=ExtractResponse)
async def extract_all(request: ExtractRequest):
    """
    Extract all URLs, documents, and content from a faculty profile page
    
    Returns:
        - profile_data: Basic profile information
        - all_urls: All URLs found on the page
        - document_links: Document links (PDF, Word, text)
        - text_content: Extracted text from profile page
        - extraction_metadata: Metadata about extraction
    """
    try:
        scraper = await get_scraper()
        result = await scraper.extract_all(request.profile_url)
        
        # Prepare JSON response first
        json_response = ExtractResponse(
            profile_data=result['profile_data'],
            all_urls=result['all_urls'],
            document_links=result['document_links'],
            text_content=result['text_content'],
            body_links_with_content=result.get('body_links_with_content', []),
            extraction_metadata=result['extraction_metadata']
        )
        
        # After JSON is prepared, save ALL content for this profile to JSON file
        try:
            from api.services.json_writer import get_json_writer
            import uuid
            
            json_writer = get_json_writer()
            
            # Get profile information
            profile_data = result.get('profile_data', {})
            profile_name = profile_data.get('name', '') or 'Unknown'
            profile_id = str(uuid.uuid4())[:8]
            
            # Collect all URLs
            all_urls = [request.profile_url]  # Start with profile URL
            body_links = result.get('body_links_with_content', [])
            for link in body_links:
                if link.get('source_url'):
                    all_urls.append(link.get('source_url'))
            
            # Combine all text content
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
            
            # Only write if we have content
            if combined_text or profile_text.get('full_text') or body_links:
                print(f"[Extract] Preparing to write - Profile: {profile_name}, URLs: {len(all_urls)}, Text parts: {len(combined_text_parts)}")
                
                # Use the new data cleaning pipeline for normalization, section extraction, and chunking
                chunks = []
                try:
                    from api.services.data_cleaning_service import get_data_cleaning_service
                    
                    # Initialize the data cleaning service with optimal chunk sizes
                    # Target: 325 words, Range: 250-400 words per chunk
                    cleaning_service = get_data_cleaning_service(
                        target_words_per_chunk=325,
                        min_words_per_chunk=250,
                        max_words_per_chunk=400
                    )
                    
                    # Process text through the complete pipeline
                    print(f"[Extract] Processing text through data cleaning pipeline...")
                    pipeline_chunks = cleaning_service.clean_and_chunk_text(
                        text=combined_text,
                        profile_url=request.profile_url,
                        section_header=""
                    )
                    
                    # Convert pipeline chunks to the format expected by json_writer
                    for idx, chunk in enumerate(pipeline_chunks):
                        chunks.append({
                            'chunk_id': chunk.get('id', f"chunk_{idx}"),
                            'chunk_index': idx,
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
                    
                    print(f"[Extract] Created {len(chunks)} chunks using data cleaning pipeline")
                    
                except Exception as pipeline_error:
                    import traceback
                    print(f"[Extract] Warning: Data cleaning pipeline failed: {str(pipeline_error)}")
                    print(traceback.format_exc())
                    print(f"[Extract] Falling back to LangChain chunker...")
                    
                    # Fallback to LangChain chunker if pipeline fails
                    try:
                        from api.services.text_chunker import get_text_chunker
                        
                        # Build source sections from body links for better section tracking
                        source_sections = []
                        current_pos = 0
                        for link in body_links:
                            if link.get('content') and link.get('processing_status') == 'success':
                                source_type = link.get('source_type', 'webpage')
                                source_url = link.get('source_url', '')
                                link_text = link.get('link_text', '') or 'Content'
                                
                                # Find section marker in combined_text
                                section_marker = f"=== {source_type.upper()}: {link_text} ({source_url}) ==="
                                section_start = combined_text.find(section_marker, current_pos)
                                if section_start >= 0:
                                    # Find where this section ends (next section or end)
                                    section_end = len(combined_text)
                                    for next_link in body_links:
                                        if next_link != link and next_link.get('content'):
                                            next_source_type = next_link.get('source_type', 'webpage')
                                            next_source_url = next_link.get('source_url', '')
                                            next_link_text = next_link.get('link_text', '') or 'Content'
                                            next_marker = f"=== {next_source_type.upper()}: {next_link_text} ({next_source_url}) ==="
                                            next_pos = combined_text.find(next_marker, section_start + len(section_marker))
                                            if next_pos >= 0 and next_pos < section_end:
                                                section_end = next_pos
                                    
                                    source_sections.append({
                                        'start': section_start,
                                        'end': section_end,
                                        'type': link_text,
                                        'source': source_type
                                    })
                                    current_pos = section_start + len(section_marker)
                        
                        # Get chunker and create chunks with section information
                        # Use LLM for section detection (will fallback to pattern if LLM unavailable)
                        use_llm_for_sections = True  # Always try LLM for better section detection
                        
                        # Chunk size 1000, overlap will be 10% (100 chars) automatically
                        chunker = get_text_chunker(chunk_size=1000, chunk_overlap=None)  # None = 10% of chunk_size
                        chunks = chunker.chunk_text_with_sections(
                            text=combined_text,
                            headings=combined_headings,
                            source_sections=source_sections if source_sections else None,
                            use_llm_for_sections=use_llm_for_sections,
                            professor_name=profile_name
                        )
                        print(f"[Extract] Created {len(chunks)} chunks with section information (LangChain fallback)")
                    except Exception as chunk_error:
                        import traceback
                        print(f"[Extract] ERROR: Failed to chunk text: {str(chunk_error)}")
                        print(traceback.format_exc())
                        # Fallback: Create basic chunks if LangChain fails
                        if combined_text:
                            # Simple fallback chunking - split by paragraphs
                            paragraphs = combined_text.split('\n\n')
                            chunk_text = ''
                            chunk_idx = 0
                            
                            # Helper function to clean chunk text (remove newlines)
                            def clean_chunk_text(text):
                                import re
                                cleaned = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                                cleaned = re.sub(r'\s+', ' ', cleaned)
                                return cleaned.strip()
                            
                            for para in paragraphs:
                                if len(chunk_text) + len(para) > 1000 and chunk_text:
                                    cleaned_text = clean_chunk_text(chunk_text)
                                    chunks.append({
                                        'chunk_id': f"chunk_{chunk_idx}",
                                        'chunk_index': chunk_idx,
                                        'text': cleaned_text,
                                        'char_count': len(cleaned_text),
                                        'word_count': len(cleaned_text.split()),
                                        'professor_name': profile_name,
                                        'section': '',
                                        'heading': '',
                                        'section_type': 'fallback',
                                        'source': 'fallback_chunking',
                                        'start_position': 0,
                                        'end_position': len(chunk_text)
                                    })
                                    chunk_idx += 1
                                    chunk_text = para + '\n\n'
                                else:
                                    chunk_text += para + '\n\n'
                            
                            # Add last chunk
                            if chunk_text.strip():
                                cleaned_text = clean_chunk_text(chunk_text)
                                chunks.append({
                                    'chunk_id': f"chunk_{chunk_idx}",
                                    'chunk_index': chunk_idx,
                                    'text': cleaned_text,
                                    'char_count': len(cleaned_text),
                                    'word_count': len(cleaned_text.split()),
                                    'professor_name': profile_name,
                                    'section': '',
                                    'heading': '',
                                    'section_type': 'fallback',
                                    'source': 'fallback_chunking',
                                    'start_position': 0,
                                    'end_position': len(chunk_text)
                                })
                            print(f"[Extract] Created {len(chunks)} fallback chunks")
                
                # Write profile content to JSON (with chunks)
                json_writer.write_profile_content(
                    profile_id=profile_id,
                    profile_name=profile_name,
                    profile_url=request.profile_url,
                    all_urls=all_urls,
                    combined_text=combined_text,
                    combined_headings=combined_headings,
                    combined_paragraphs=combined_paragraphs,
                    cleaning_status='pending',
                    chunks=chunks
                )
                
                # Save JSON file
                json_writer.save()
                print(f"[Extract] SUCCESS: Saved profile content to JSON - ID: {profile_id}, Name: {profile_name}, Total URLs: {len(all_urls)}, Combined text length: {len(combined_text)}, Chunks: {len(chunks)}")
            else:
                print(f"[Extract] WARNING: No content to save for profile: {profile_name}")
            
        except Exception as e:
            import traceback
            print(f"[Extract] Warning: Failed to save to JSON: {str(e)}")
            print(traceback.format_exc())
            # Don't fail the request if JSON save fails
        
        return json_response
    
    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        import traceback
        print(f"Extraction error: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Extraction failed: {error_msg}")



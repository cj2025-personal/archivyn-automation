"""
Process endpoint - processes documents and webpages
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from api.services.document_processor import get_document_processor
from api.services.scraper import get_scraper
from api.services.llm_extractor import get_llm_extractor

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    resource_url: str
    resource_type: str  # "document" or "webpage"
    category: str  # "document", "personal_website", "academic_profile", etc.
    link_text: Optional[str] = ""
    source: str = "profile_page"


class ProcessResponse(BaseModel):
    raw_content: str
    structured_data: Dict
    metadata: Dict
    resource_type: str
    category: str
    url: Optional[str] = None  # Added for convenience


@router.post("/process-resource", response_model=ProcessResponse)
async def process_resource(request: ProcessRequest):
    """
    Process a resource (document or webpage) and extract structured data
    
    For documents: Downloads and extracts text from PDF/Word/text files
    For webpages: Scrapes content and extracts structured data
    
    Returns:
        - raw_content: Full extracted text
        - structured_data: LLM-extracted structured data
        - metadata: Processing metadata
    """
    try:
        raw_content = ""
        metadata = {}
        
        if request.resource_type == "document":
            # Process document (PDF, Word, text)
            doc_processor = get_document_processor()
            doc_result = doc_processor.process_document(request.resource_url)
            
            raw_content = doc_result['content']
            metadata = {
                'file_type': doc_result['file_type'],
                'word_count': doc_result['word_count'],
                'file_size': doc_result.get('file_size', 0),
                'document_metadata': doc_result.get('metadata', {})
            }
        
        elif request.resource_type == "webpage":
            # Scrape webpage
            scraper = await get_scraper()
            webpage_result = await scraper.scrape_webpage(request.resource_url)
            
            raw_content = webpage_result['text_content'].get('full_text', '')
            metadata = {
                'title': webpage_result.get('title', ''),
                'url': request.resource_url
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported resource type: {request.resource_type}"
            )
        
        # Extract structured data using LLM
        llm_extractor = get_llm_extractor(use_ollama=True)
        source_type = "cv" if request.category == "document" else "website"
        structured_data = llm_extractor.extract_from_text(raw_content, source_type)
        
        return ProcessResponse(
            raw_content=raw_content[:10000],  # Limit raw content in response
            structured_data=structured_data,
            metadata=metadata,
            resource_type=request.resource_type,
            category=request.category,
            url=request.resource_url
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resource processing failed: {str(e)}"
        )


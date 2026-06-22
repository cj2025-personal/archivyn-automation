"""
Batch cleaning endpoint - clean raw content stored in JSON
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from api.services.batch_cleaner import BatchCleaner, clean_all_pending_content

router = APIRouter(prefix="/clean", tags=["cleaning"])


class CleanRequest(BaseModel):
    json_file_path: Optional[str] = "extracted_content.json"
    use_llm: bool = True
    method: str = "auto"  # 'auto', 'regex', or 'llm'
    content_id: Optional[str] = None  # If provided, clean only this item


class CleanResponse(BaseModel):
    success: bool
    message: str
    stats: Optional[Dict] = None


@router.post("/batch", response_model=CleanResponse)
async def batch_clean(request: CleanRequest):
    """
    Clean all pending raw content stored in JSON
    
    Args:
        request: CleanRequest with cleaning options
    
    Returns:
        CleanResponse with cleaning statistics
    """
    try:
        if request.content_id:
            # Clean single item
            cleaner = BatchCleaner(request.json_file_path)
            success = cleaner.clean_single_item(request.content_id, method=request.method)
            
            if success:
                return CleanResponse(
                    success=True,
                    message=f"Successfully cleaned content ID: {request.content_id}",
                    stats={'cleaned': 1, 'total': 1}
                )
            else:
                return CleanResponse(
                    success=False,
                    message=f"Failed to clean content ID: {request.content_id}",
                    stats={'cleaned': 0, 'total': 1}
                )
        else:
            # Clean all pending
            stats = clean_all_pending_content(
                json_file_path=request.json_file_path,
                use_llm=request.use_llm,
                method=request.method
            )
            
            return CleanResponse(
                success=True,
                message=f"Batch cleaning complete: {stats['cleaned']}/{stats['total']} items cleaned",
                stats=stats
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch cleaning: {str(e)}"
        )


@router.get("/pending-count")
async def get_pending_count(json_file_path: str = "extracted_content.json"):
    """
    Get count of pending content items
    
    Returns:
        Dictionary with pending count
    """
    try:
        from api.services.json_writer import get_json_writer
        json_writer = get_json_writer(json_file_path)
        pending_items = json_writer.get_pending_content()
        
        return {
            "pending_count": len(pending_items),
            "json_file": json_file_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting pending count: {str(e)}"
        )


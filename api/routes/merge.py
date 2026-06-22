"""
Merge and store endpoint - merges data from all sources and writes to Excel
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from api.services.excel_writer import get_excel_writer
import uuid
from datetime import datetime

router = APIRouter(prefix="/api", tags=["merge"])


class ProcessedResource(BaseModel):
    url: str
    category: str
    resource_type: str
    structured_data: Dict
    raw_content: Optional[str] = None
    metadata: Optional[Dict] = None


class MergeRequest(BaseModel):
    profile_url: str
    profile_data: Dict
    processed_resources: List[ProcessedResource]


class MergeResponse(BaseModel):
    profile_id: str
    merged_data: Dict
    excel_row_written: bool
    sources_used: List[str]
    data_quality_score: int


@router.post("/merge-and-store", response_model=MergeResponse)
async def merge_and_store(request: MergeRequest):
    """
    Merge data from profile page and all processed resources,
    then write to Excel file
    
    Returns:
        - profile_id: Unique ID for this profile
        - merged_data: Combined data from all sources
        - excel_row_written: Whether data was written successfully
        - sources_used: List of sources that contributed data
        - data_quality_score: Quality score (0-100)
    """
    try:
        # Generate unique profile ID
        profile_id = str(uuid.uuid4())[:8]
        
        # Merge data from all sources
        merged_data = merge_professor_data(
            request.profile_data,
            request.processed_resources
        )
        
        # Determine sources used
        sources_used = ["profile"]
        if any(r.category == "document" for r in request.processed_resources):
            sources_used.append("document")
        if any(r.category == "personal_website" for r in request.processed_resources):
            sources_used.append("website")
        if any(r.category == "academic_profile" for r in request.processed_resources):
            sources_used.append("academic_profile")
        
        merged_data['sources_used'] = ', '.join(sources_used)
        
        # Calculate quality score
        quality_score = calculate_quality_score(merged_data)
        merged_data['data_quality_score'] = quality_score
        merged_data['profile_id'] = profile_id
        
        # Write to Excel
        excel_writer = get_excel_writer()
        
        # Write faculty profile
        metadata = {
            'extraction_status': 'Complete',
            'total_resources_found': len(request.processed_resources),
            'resources_processed': len(request.processed_resources),
            'sources_used': ', '.join(sources_used)
        }
        excel_writer.write_faculty_profile(profile_id, request.profile_data, metadata)
        
        # Write extracted resources
        for resource in request.processed_resources:
            resource_data = {
                'url': resource.url,
                'resource_type': resource.resource_type,
                'category': resource.category,
                'link_text': '',
                'file_type': resource.metadata.get('file_type', '') if resource.metadata else '',
                'word_count': resource.metadata.get('word_count', 0) if resource.metadata else 0
            }
            excel_writer.write_extracted_resource(profile_id, resource_data)
        
        # Write merged data
        excel_writer.write_merged_data(profile_id, merged_data)
        
        # Write detailed data
        if merged_data.get('publications'):
            excel_writer.write_publications(
                profile_id,
                merged_data['publications'],
                'merged'
            )
        
        if merged_data.get('education'):
            excel_writer.write_education(
                profile_id,
                merged_data['education'],
                'merged'
            )
        
        if merged_data.get('awards'):
            excel_writer.write_awards(
                profile_id,
                merged_data['awards'],
                'merged'
            )
        
        if merged_data.get('expertise'):
            excel_writer.write_expertise(
                profile_id,
                merged_data['expertise'],
                'merged'
            )
        
        if merged_data.get('experience'):
            excel_writer.write_experience(
                profile_id,
                merged_data['experience'],
                'merged'
            )
        
        # Save Excel file
        excel_writer.save()
        
        return MergeResponse(
            profile_id=profile_id,
            merged_data=merged_data,
            excel_row_written=True,
            sources_used=sources_used,
            data_quality_score=quality_score
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Merge and store failed: {str(e)}"
        )


def merge_professor_data(profile_data: Dict, processed_resources: List[ProcessedResource]) -> Dict:
    """Merge data from profile and all resources"""
    merged = {
        'name': profile_data.get('name', ''),
        'university': profile_data.get('university', ''),
        'department': profile_data.get('department', ''),
        'email': profile_data.get('email', ''),
        'position': profile_data.get('position', ''),
        'bio': profile_data.get('bio', '') or profile_data.get('full_text', ''),
        'education': [],
        'publications': [],
        'awards': [],
        'expertise': [],
        'experience': [],
        'milestones': []
    }
    
    # Collect data from all resources
    all_publications = []
    all_education = []
    all_awards = []
    all_expertise = []
    all_experience = []
    all_milestones = []
    bios = []
    
    # Add profile bio if available
    if merged['bio']:
        bios.append(merged['bio'])
    
    # Process each resource
    for resource in processed_resources:
        structured = resource.structured_data
        
        # Merge bio (use longest)
        if structured.get('bio'):
            bios.append(structured['bio'])
        
        # Collect publications
        if structured.get('publications'):
            all_publications.extend(structured['publications'])
        
        # Collect education
        if structured.get('education'):
            all_education.extend(structured['education'])
        
        # Collect awards
        if structured.get('awards'):
            all_awards.extend(structured['awards'])
        
        # Collect expertise
        if structured.get('expertise'):
            all_expertise.extend(structured['expertise'])
        
        # Collect experience
        if structured.get('experience'):
            all_experience.extend(structured['experience'])
        
        # Collect milestones
        if structured.get('milestones'):
            all_milestones.extend(structured['milestones'])
    
    # Use longest bio
    if bios:
        merged['bio'] = max(bios, key=len)
    
    # Deduplicate and merge lists
    merged['publications'] = deduplicate_publications(all_publications)
    merged['education'] = deduplicate_education(all_education)
    merged['awards'] = deduplicate_awards(all_awards)
    merged['expertise'] = deduplicate_expertise(all_expertise)
    merged['experience'] = deduplicate_experience(all_experience)
    merged['milestones'] = list(set(all_milestones))  # Simple deduplication
    
    return merged


def deduplicate_publications(publications: List[Dict]) -> List[Dict]:
    """Remove duplicate publications based on title"""
    seen_titles = set()
    unique = []
    
    for pub in publications:
        title = pub.get('title', '').lower().strip() if isinstance(pub, dict) else str(pub).lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique.append(pub)
    
    return unique


def deduplicate_education(education: List) -> List:
    """Remove duplicate education entries"""
    seen = set()
    unique = []
    
    for edu in education:
        if isinstance(edu, dict):
            key = f"{edu.get('degree', '')}-{edu.get('institution', '')}"
        else:
            key = str(edu)
        
        if key not in seen:
            seen.add(key)
            unique.append(edu)
    
    return unique


def deduplicate_awards(awards: List) -> List:
    """Remove duplicate awards"""
    seen = set()
    unique = []
    
    for award in awards:
        if isinstance(award, dict):
            key = award.get('name', '').lower()
        else:
            key = str(award).lower()
        
        if key and key not in seen:
            seen.add(key)
            unique.append(award)
    
    return unique


def deduplicate_expertise(expertise: List) -> List:
    """Remove duplicate expertise areas"""
    seen = set()
    unique = []
    
    for exp in expertise:
        if isinstance(exp, dict):
            key = exp.get('area', '') or exp.get('expertise_area', '')
        else:
            key = str(exp)
        
        key_lower = key.lower().strip()
        if key_lower and key_lower not in seen:
            seen.add(key_lower)
            unique.append(exp)
    
    return unique


def deduplicate_experience(experience: List) -> List:
    """Remove duplicate experience entries"""
    seen = set()
    unique = []
    
    for exp in experience:
        if isinstance(exp, dict):
            key = f"{exp.get('position', '')}-{exp.get('institution', '')}"
        else:
            key = str(exp)
        
        if key not in seen:
            seen.add(key)
            unique.append(exp)
    
    return unique


def calculate_quality_score(data: Dict) -> int:
    """Calculate data quality score (0-100)"""
    score = 0
    
    # Bio (20 points)
    if data.get('bio') and len(data['bio']) > 100:
        score += 20
    elif data.get('bio'):
        score += 10
    
    # Publications (20 points)
    pub_count = len(data.get('publications', []))
    if pub_count > 10:
        score += 20
    elif pub_count > 5:
        score += 15
    elif pub_count > 0:
        score += 10
    
    # Education (20 points)
    if len(data.get('education', [])) > 0:
        score += 20
    
    # Expertise (20 points)
    exp_count = len(data.get('expertise', []))
    if exp_count > 5:
        score += 20
    elif exp_count > 0:
        score += 15
    
    # Awards/Experience (20 points)
    if len(data.get('awards', [])) > 0 or len(data.get('experience', [])) > 0:
        score += 20
    
    return min(score, 100)




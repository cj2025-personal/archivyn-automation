"""
Script to upload chunks from a single profile to Pinecone vector database
Takes a profile_id (folder name) and uploads all chunks from that profile
"""
import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import services
from api.services.vector_db import get_vector_db
from api.services.embeddings import get_embeddings_service
from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION


def extract_professor_name(chunks_data: Dict[str, Any]) -> Optional[str]:
    """
    Extract professor name from chunks data
    Tries multiple strategies to find the most accurate full name
    
    Args:
        chunks_data: The full chunks JSON data for a profile
        
    Returns:
        Professor name if found, None otherwise
    """
    sections = chunks_data.get('sections', {})
    all_text = []  # Collect all text for cross-validation
    
    # Collect text from all sections for analysis
    for section_name, section_chunks in sections.items():
        for chunk in section_chunks:
            all_text.append(chunk.get('text', ''))
    
    combined_text = ' '.join(all_text)
    
    # Strategy 1: Look for "Dr. LastName" or "Dr LastName" patterns to identify last name
    last_name_patterns = []
    dr_pattern = re.findall(r'Dr\.?\s+([A-Z][a-z]+)', combined_text)
    if dr_pattern:
        # Get the most common last name mentioned after "Dr."
        from collections import Counter
        last_name_counts = Counter(dr_pattern)
        if last_name_counts:
            most_common_last = last_name_counts.most_common(1)[0][0]
            last_name_patterns.append(most_common_last)
    
    # Strategy 2: Try Contact section first (usually has full name)
    contact_chunks = sections.get('Contact', [])
    contact_names = []
    if contact_chunks:
        contact_text = contact_chunks[0].get('text', '')
        # Look for full names - capture up to 4 words (First Middle Middle Last)
        # Pattern: "FirstName [MiddleName(s)] LastName"
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',  # Full name with middle names
            r'([A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+)',  # First M. Last
            r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Middle Last
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, contact_text)
            if matches:
                contact_names.extend(matches)
    
    # Strategy 3: Try Biography section (often has full name at start)
    bio_chunks = sections.get('Biography', [])
    bio_names = []
    if bio_chunks:
        bio_text = bio_chunks[0].get('text', '')
        # Look for "FirstName MiddleName LastName is..." pattern
        # This is the most reliable pattern - name at start of biography
        bio_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is',  # "Name is..."
            r'^([A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+)\s+is',  # "First M. Last is..."
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\s+is',  # "First Middle Last is..."
        ]
        for pattern in bio_patterns:
            match = re.search(pattern, bio_text)
            if match:
                bio_names.append(match.group(1).strip())
    
    # Strategy 4: Cross-validate with last name from "Dr. LastName" patterns
    candidate_names = []
    
    # Combine all found names
    all_candidates = contact_names + bio_names
    
    # If we found a last name from "Dr. LastName" pattern, validate candidates
    if last_name_patterns:
        last_name = last_name_patterns[0]
        # Filter candidates that end with the identified last name
        for name in all_candidates:
            name_parts = name.split()
            if name_parts and name_parts[-1] == last_name:
                candidate_names.append(name)
    
    # If no cross-validation possible, use all candidates
    if not candidate_names:
        candidate_names = all_candidates
    
    # Select the best candidate:
    # 1. Prefer names that end with a last name found in "Dr. LastName" pattern
    # 2. Prefer longer names (more complete)
    # 3. Prefer names from Biography (usually more accurate)
    if candidate_names:
        # Sort by: length (longer = more complete), then prefer bio names
        def score_name(name):
            score = len(name.split())  # More words = better
            if name in bio_names:
                score += 10  # Prefer bio names
            if last_name_patterns and name.split()[-1] == last_name_patterns[0]:
                score += 5  # Prefer names matching Dr. pattern
            return score
        
        best_name = max(candidate_names, key=score_name)
        return best_name.strip()
    
    # Fallback: If no good match, try simple pattern matching
    if bio_chunks:
        bio_text = bio_chunks[0].get('text', '')
        # Simple fallback: first 3 capitalized words
        simple_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', bio_text)
        if simple_match:
            return simple_match.group(1).strip()
    
    return None


def get_profile_name_from_json(profile_id: str, output_dir: str) -> Optional[str]:
    """
    Get professor name from profile JSON file (which uses Excel name)
    
    Args:
        profile_id: Profile ID
        output_dir: Base output directory (e.g., "output")
        
    Returns:
        Professor name if found, None otherwise
    """
    output_path = Path(output_dir)
    profile_json_file = output_path / "profiles" / profile_id / f"{profile_id}.json"
    
    if profile_json_file.exists():
        try:
            with open(profile_json_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            name = profile_data.get('name', '')
            if name and name.strip():
                return name.strip()
        except Exception as e:
            print(f"[Loading] ⚠️ Could not read profile JSON: {str(e)}")
    
    return None


def load_profile_chunks(profile_id: str, chunks_dir: str, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load chunks from a single profile
    
    Args:
        profile_id: Profile ID (folder name)
        chunks_dir: Directory containing chunked profiles
        output_dir: Base output directory to look for profile JSON (default: parent of chunks_dir)
        
    Returns:
        List of chunks with their metadata, including professor_id and professor_name
    """
    chunks_dir_path = Path(chunks_dir)
    profile_dir = chunks_dir_path / profile_id
    chunks_file = profile_dir / "chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    print(f"[Loading] Loading chunks from: {chunks_file}")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Verify profile_id matches
        file_profile_id = data.get('profile_id', '')
        if file_profile_id != profile_id:
            print(f"[Loading] ⚠️ Warning: Profile ID mismatch. File has '{file_profile_id}', expected '{profile_id}'")
        
        sections = data.get('sections', {})
        
        # Try to get name from profile JSON file first (Excel name - more accurate)
        professor_name = None
        if output_dir:
            professor_name = get_profile_name_from_json(profile_id, output_dir)
            if professor_name:
                print(f"[Loading] ✅ Using name from profile JSON (Excel): {professor_name}")
        
        # Fallback to extraction from chunks if not found in JSON
        if not professor_name:
            professor_name = extract_professor_name(data)
            if professor_name:
                print(f"[Loading] ✅ Extracted professor name from chunks: {professor_name}")
            else:
                print(f"[Loading] ⚠️ Could not extract professor name")
        
        # Extract all chunks from all sections
        all_chunks = []
        for section_name, section_chunks in sections.items():
            for chunk in section_chunks:
                chunk_data = {
                    'profile_id': profile_id,
                    'professor_id': profile_id,  # Use profile_id as professor_id
                    'professor_name': professor_name,  # Extracted name or None
                    'section': chunk.get('section', section_name),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'order': chunk.get('order', 0),
                    'text': chunk.get('text', ''),
                }
                all_chunks.append(chunk_data)
        
        print(f"[Loading] ✅ Loaded {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
        
    except Exception as e:
        print(f"[Loading] ❌ Error loading chunks: {str(e)}")
        raise


def generate_vector_id(profile_id: str, chunk_id: str) -> str:
    """
    Generate a unique vector ID for a chunk
    
    Args:
        profile_id: Profile ID
        chunk_id: Chunk ID
        
    Returns:
        Unique vector ID
    """
    # Use chunk_id if available, otherwise combine profile_id and section/order
    if chunk_id:
        return f"chunk_{chunk_id}"
    else:
        return f"profile_{profile_id}_{chunk_id}"


def upload_chunks_to_pinecone(chunks: List[Dict[str, Any]], batch_size: int = 100):
    """
    Generate embeddings and upload chunks to Pinecone
    
    Args:
        chunks: List of chunk dictionaries
        batch_size: Number of chunks to process in each batch
    """
    print(f"\n[Setup] Initializing services...")
    print(f"[Setup] Index: {INDEX_NAME}")
    print(f"[Setup] Dimension: {INDEX_DIMENSION}")
    print(f"[Setup] Model: text-embedding-3-small")
    
    # Initialize services
    try:
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        embeddings_service = get_embeddings_service()
        
        print(f"[Setup] ✅ Connected to Pinecone index: {INDEX_NAME}")
        print(f"[Setup] ✅ Initialized embeddings service: {embeddings_service.model_name}")
    except Exception as e:
        print(f"[Setup] ❌ Error initializing services: {str(e)}")
        raise
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.get('text', '').strip()]
    print(f"\n[Processing] Processing {len(valid_chunks)} valid chunks (skipped {len(chunks) - len(valid_chunks)} empty chunks)")
    
    if not valid_chunks:
        print("[Processing] ❌ No valid chunks to process!")
        return
    
    # Process in batches
    total_batches = (len(valid_chunks) + batch_size - 1) // batch_size
    successful_uploads = 0
    failed_uploads = 0
    
    print(f"\n[Processing] Processing {len(valid_chunks)} chunks in {total_batches} batches...")
    
    for batch_idx in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} chunks...")
        
        try:
            # Extract texts for embedding generation
            texts = [chunk['text'] for chunk in batch]
            
            # Generate embeddings in batch
            print(f"[Batch {batch_num}] Generating embeddings...")
            embeddings = embeddings_service.embed_batch(texts, batch_size=len(texts))
            
            if len(embeddings) != len(batch):
                print(f"[Batch {batch_num}] ⚠️ Warning: Got {len(embeddings)} embeddings for {len(batch)} chunks")
            
            # Prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(batch):
                if i >= len(embeddings):
                    print(f"[Batch {batch_num}] ⚠️ Skipping chunk {i} - no embedding available")
                    failed_uploads += 1
                    continue
                
                embedding = embeddings[i]
                
                # Validate embedding
                if len(embedding) != INDEX_DIMENSION:
                    print(f"[Batch {batch_num}] ⚠️ Skipping chunk {i} - dimension mismatch: {len(embedding)} != {INDEX_DIMENSION}")
                    failed_uploads += 1
                    continue
                
                if all(v == 0.0 for v in embedding):
                    print(f"[Batch {batch_num}] ⚠️ Skipping chunk {i} - zero vector")
                    failed_uploads += 1
                    continue
                
                # Generate vector ID
                vector_id = generate_vector_id(chunk['profile_id'], chunk['chunk_id'])
                
                # Prepare metadata
                metadata = {
                    'profile_id': chunk['profile_id'],
                    'professor_id': chunk.get('professor_id', chunk['profile_id']),  # Unique identifier for professor
                    'section': chunk['section'],
                    'chunk_id': chunk['chunk_id'],
                    'order': chunk['order'],
                    'text': chunk['text'],  # Store full text in metadata
                    'content_type': 'profile_chunk',
                }
                
                # Add professor_name if available
                if chunk.get('professor_name'):
                    metadata['professor_name'] = chunk['professor_name']
                
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upload to Pinecone
            if vectors_to_upsert:
                print(f"[Batch {batch_num}] Uploading {len(vectors_to_upsert)} vectors to Pinecone...")
                
                # Upsert in smaller sub-batches (Pinecone recommends max 100 vectors per upsert)
                pinecone_batch_size = 100
                for sub_batch_idx in range(0, len(vectors_to_upsert), pinecone_batch_size):
                    sub_batch = vectors_to_upsert[sub_batch_idx:sub_batch_idx + pinecone_batch_size]
                    vector_db.index.upsert(vectors=sub_batch)
                
                successful_uploads += len(vectors_to_upsert)
                print(f"[Batch {batch_num}] ✅ Successfully uploaded {len(vectors_to_upsert)} vectors")
            else:
                print(f"[Batch {batch_num}] ⚠️ No valid vectors to upload in this batch")
            
            # Small delay to avoid rate limiting
            if batch_num < total_batches:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[Batch {batch_num}] ❌ Error processing batch: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_uploads += len(batch)
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"[Summary] Upload Complete!")
    print(f"[Summary] Successful uploads: {successful_uploads}")
    print(f"[Summary] Failed uploads: {failed_uploads}")
    print(f"[Summary] Total chunks processed: {len(valid_chunks)}")
    print(f"{'='*60}")
    
    # Verify by checking index stats
    try:
        stats = vector_db.index.describe_index_stats()
        print(f"\n[Verification] Pinecone index stats:")
        print(f"[Verification] Total vectors: {stats.total_vector_count}")
        print(f"[Verification] Namespaces: {stats.namespaces}")
    except Exception as e:
        print(f"[Verification] Could not retrieve index stats: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Upload chunks from a single profile to Pinecone'
    )
    parser.add_argument(
        'profile_id',
        type=str,
        help='Profile ID (folder name) to upload'
    )
    parser.add_argument(
        '--chunks-dir',
        type=str,
        default=None,
        help='Path to chunked profiles directory (default: output/chunked_profiles)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base output directory to look for profile JSON files (default: output)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Determine chunks directory
    if args.chunks_dir:
        chunks_dir = args.chunks_dir
    else:
        chunks_dir = os.path.join(os.path.dirname(__file__), "output", "chunked_profiles")
    
    # Determine output directory (for profile JSON files)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    
    profile_id = args.profile_id
    
    print("="*60)
    print("Pinecone Single Profile Upload Script")
    print("="*60)
    print(f"Profile ID: {profile_id}")
    print(f"Chunks directory: {chunks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Index: {INDEX_NAME}")
    print(f"Dimension: {INDEX_DIMENSION}")
    print(f"Model: text-embedding-3-small")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(chunks_dir):
        print(f"❌ Error: Chunks directory not found: {chunks_dir}")
        return
    
    # Load chunks for this profile
    try:
        chunks = load_profile_chunks(profile_id, chunks_dir, output_dir=output_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        return
    except Exception as e:
        print(f"❌ Error loading chunks: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    if not chunks:
        print("❌ No chunks found to upload!")
        return
    
    # Upload to Pinecone
    upload_chunks_to_pinecone(chunks, batch_size=args.batch_size)


if __name__ == "__main__":
    main()


"""
Script to run the Profile Chunking Pipeline on existing profile JSON files.

Usage:
    python run_chunking_pipeline.py --input-dir E:/stage_cache --output-dir output/chunked_profiles
    python run_chunking_pipeline.py --profile-id <uuid> --input-dir E:/stage_cache
"""
import json
import argparse
import sys
from pathlib import Path
from typing import Optional

from profile_chunking_pipeline import ProfileChunkingPipeline


def find_stage_cache_dir() -> Optional[str]:
    """Try to find stage_cache directory"""
    project_root = Path(__file__).parent
    possible_locations = [
        Path("E:/stage_cache"),
        Path("stage_cache"),
        project_root / "stage_cache",
    ]
    
    for loc in possible_locations:
        if loc.exists() and loc.is_dir():
            return str(loc.resolve())
    return None


def load_profile_json(filepath: Path) -> Optional[dict]:
    """Load profile JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle nested profile structure
            if 'profile' in data:
                return data['profile']
            return data
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None


def process_single_profile(profile_id: str, input_dir: str, pipeline: ProfileChunkingPipeline):
    """Process a single profile by ID"""
    input_path = Path(input_dir)
    
    # Find JSON file (could be stage2_*.json or just *.json)
    json_files = list(input_path.glob(f"stage2_{profile_id}.json"))
    if not json_files:
        json_files = list(input_path.glob(f"{profile_id}.json"))
    
    if not json_files:
        print(f"[ERROR] Profile {profile_id} not found in {input_dir}")
        return False
    
    json_file = json_files[0]
    profile_data = load_profile_json(json_file)
    
    if not profile_data:
        return False
    
    # Check if profile has CV
    has_cv = profile_data.get('has_cv', False)
    if not has_cv:
        print(f"[SKIP] Profile {profile_id} does not have CV (has_cv: false)")
        return False
    
    clean_text = profile_data.get('clean_text', '')
    if not clean_text or not clean_text.strip():
        print(f"[WARNING] Profile {profile_id} has CV but no clean_text, skipping")
        return False
    
    print(f"[INFO] Processing profile {profile_id} ({len(clean_text)} chars)")
    try:
        pipeline.process_profile(profile_id, clean_text)
        print(f"[SUCCESS] ✅ Processed profile {profile_id}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process profile {profile_id}: {e}")
        return False


def process_all_profiles(input_dir: str, pipeline: ProfileChunkingPipeline, limit: Optional[int] = None, only_with_cv: bool = True):
    """Process all profiles in directory
    
    Args:
        input_dir: Directory containing JSON files
        pipeline: Chunking pipeline instance
        limit: Optional limit on number of profiles to process
        only_with_cv: If True, only process profiles with has_cv=True and clean_text
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob("stage2_*.json"))
    
    if not json_files:
        json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] No JSON files found in {input_dir}")
        return
    
    total = len(json_files)
    
    # Filter for profiles with CVs if requested
    if only_with_cv:
        print(f"[INFO] Filtering for profiles with CVs...")
        profiles_with_cv = []
        for json_file in json_files:
            profile_data = load_profile_json(json_file)
            if not profile_data:
                continue
            
            has_cv = profile_data.get('has_cv', False)
            clean_text = profile_data.get('clean_text', '')
            
            if has_cv and clean_text and clean_text.strip():
                profiles_with_cv.append(json_file)
        
        json_files = profiles_with_cv
        print(f"[INFO] Found {len(json_files)} profiles with CVs out of {total} total profiles")
    
    if limit:
        json_files = json_files[:limit]
        print(f"[INFO] Processing {len(json_files)} profiles (limit: {limit})")
    else:
        print(f"[INFO] Processing {len(json_files)} profiles")
    
    success = 0
    failed = 0
    skipped = 0
    
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n[{idx}/{len(json_files)}] Processing: {json_file.name}")
        
        profile_data = load_profile_json(json_file)
        if not profile_data:
            failed += 1
            continue
        
        # Extract profile ID
        profile_id = profile_data.get('profile_id') or profile_data.get('id', '')
        if not profile_id:
            # Try to extract from filename
            filename = json_file.stem
            if filename.startswith('stage2_'):
                profile_id = filename.replace('stage2_', '')
            else:
                profile_id = filename
        
        # Check if profile has CV (if filtering is enabled, this should already be true)
        has_cv = profile_data.get('has_cv', False)
        clean_text = profile_data.get('clean_text', '')
        
        if not has_cv:
            print(f"[SKIP] Profile {profile_id} does not have CV")
            skipped += 1
            continue
        
        if not clean_text or not clean_text.strip():
            print(f"[SKIP] Profile {profile_id} has CV but no clean_text")
            skipped += 1
            continue
        
        try:
            pipeline.process_profile(profile_id, clean_text)
            success += 1
            print(f"[SUCCESS] ✅ Profile {profile_id}")
        except Exception as e:
            print(f"[ERROR] ❌ Profile {profile_id}: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"[SUMMARY]")
    print(f"  Total: {len(json_files)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Profile Chunking Pipeline on JSON files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        help='Directory containing profile JSON files (default: auto-detect stage_cache)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/chunked_profiles',
        help='Output directory for chunked profiles (default: output/chunked_profiles)'
    )
    parser.add_argument(
        '--profile-id',
        type=str,
        default=None,
        help='Process single profile by ID'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of profiles to process'
    )
    parser.add_argument(
        '--all-profiles',
        action='store_true',
        help='Process all profiles (not just those with CVs)'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai'],
        help='LLM provider (default: ollama)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='mistral:7b',
        help='LLM model name (default: mistral:7b for Ollama)'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama base URL (default: http://localhost:11434)'
    )
    
    args = parser.parse_args()
    
    # Determine input directory
    input_dir = args.input_dir
    if not input_dir:
        stage_cache = find_stage_cache_dir()
        if stage_cache:
            input_dir = stage_cache
            print(f"[INFO] Auto-detected stage_cache: {input_dir}")
        else:
            print("[ERROR] No input directory specified and stage_cache not found")
            sys.exit(1)
    
    # Initialize pipeline
    print(f"[INFO] Initializing pipeline...")
    print(f"  LLM Provider: {args.llm_provider}")
    print(f"  LLM Model: {args.llm_model}")
    print(f"  Output Dir: {args.output_dir}")
    
    pipeline = ProfileChunkingPipeline(
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_base_url=args.ollama_url
    )
    
    # Process profiles
    if args.profile_id:
        # Single profile
        process_single_profile(args.profile_id, input_dir, pipeline)
    else:
        # All profiles (filter for CVs by default)
        process_all_profiles(
            input_dir, 
            pipeline, 
            limit=args.limit,
            only_with_cv=not args.all_profiles
        )


if __name__ == '__main__':
    main()


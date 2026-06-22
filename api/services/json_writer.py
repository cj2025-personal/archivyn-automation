"""
JSON writer service for storing extracted data
Creates and updates JSON files with structured data
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from api.utils.llm_text_cleaner import get_llm_text_cleaner
from api.utils.text_cleaner import get_text_cleaner
from api.services.data_cleaning_service import get_data_cleaning_service


class JSONWriter:
    """Write extracted data to JSON files"""
    
    def __init__(self, json_file_path: str = "extracted_content.json"):
        # Convert to absolute path to ensure file is created in the right location
        if not os.path.isabs(json_file_path):
            # Get the project root directory (parent of api folder)
            # __file__ is api/services/json_writer.py, so:
            # - dirname(__file__) = api/services
            # - dirname(api/services) = api
            # - dirname(api) = project root
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            api_dir = os.path.dirname(current_file_dir)  # api/services -> api
            project_root = os.path.dirname(api_dir)  # api -> project root
            self.json_file_path = os.path.join(project_root, json_file_path)
        else:
            self.json_file_path = json_file_path
        
        self.data: Dict = {}
        print(f"[JSONWriter] Initialized with file path: {self.json_file_path}")
        self._load_data()
    
    def _load_data(self):
        """Load existing data from JSON file"""
        if os.path.exists(self.json_file_path):
            try:
                # Use utf-8-sig to gracefully handle files that were written with a BOM
                # (e.g., via PowerShell's default Set-Content behavior on Windows).
                with open(self.json_file_path, 'r', encoding='utf-8-sig') as f:
                    self.data = json.load(f)
                print(f"[JSONWriter] Loaded existing data from: {self.json_file_path}")
                print(f"[JSONWriter] Found {len(self.data.get('profiles', []))} existing profiles")
            except json.JSONDecodeError as e:
                print(f"[JSONWriter] JSON file is invalid or empty: {str(e)}. Creating new file...")
                self.data = {
                    'profiles': [],
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            except Exception as e:
                print(f"[JSONWriter] Error loading existing JSON file: {str(e)}. Creating new file...")
                import traceback
                traceback.print_exc()
                self.data = {
                    'profiles': [],
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
        else:
            print(f"[JSONWriter] Creating new JSON file: {self.json_file_path}")
            self.data = {
                'profiles': [],
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
    
    def write_profile_content(self, profile_id: str, profile_name: str, profile_url: str, 
                             all_urls: List[str], combined_text: str, 
                             combined_headings: List[str] = None, combined_paragraphs: List[str] = None,
                             cleaned_text: str = None, cleaning_status: str = 'pending',
                             cleaning_method: str = None, chunks: List[Dict] = None,
                             enforce_llm_cleaning: bool = True) -> None:
        """
        Write all content for a single profile
        
        Args:
            profile_id: Unique identifier for this profile
            profile_name: Name of the profile
            profile_url: Main profile URL
            all_urls: List of all URLs scraped (profile + documents + webpages)
            combined_text: Combined raw text from all sources
            combined_headings: Combined headings from all sources
            combined_paragraphs: Combined paragraphs from all sources
            cleaned_text: Cleaned text (optional, can be added later)
            cleaning_status: Status of cleaning (pending/cleaned/failed)
            cleaning_method: Method used for cleaning (regex/llm/manual)
            chunks: List of text chunks with section information (optional)
        """
        # Mandatory cleaning pass (LLM with regex fallback) unless explicitly disabled
        cleaning_method_final = cleaning_method or ''
        cleaned_final = cleaned_text or ''
        cleaned_chunks: List[Dict] = chunks or []

        # Single-pass cleanup + fixed-size chunking with overlap
        base_text = combined_text or cleaned_final or ''
        if base_text:
            try:
                llm_cleaner = get_llm_text_cleaner() if enforce_llm_cleaning else None
                if llm_cleaner:
                    base_text = llm_cleaner.clean_text(base_text, use_chunking=True)
                    cleaning_method_final = 'llm_first_pass'
                else:
                    text_cleaner = get_text_cleaner()
                    base_text = text_cleaner.clean_text(base_text, aggressive=True, use_llm=False)
                    cleaning_method_final = 'regex_first_pass'
            except Exception as e:
                print(f"[JSONWriter] Warning: initial cleaning failed ({e}), using raw text")
            
            try:
                cleaning_service = get_data_cleaning_service(
                    target_words_per_chunk=325,
                    min_words_per_chunk=250,
                    max_words_per_chunk=400,
                    use_llm_cleaning=False  # avoid repeated OpenAI calls; chunking stays fixed-size
                )
                cleaned_chunks = cleaning_service.clean_and_chunk_text(
                    text=base_text,
                    profile_url=profile_url,
                    section_header=""
                )
                if cleaned_chunks:
                    cleaned_final = " ".join([c.get("text", "") for c in cleaned_chunks]).strip()
                    cleaning_method_final = f"{cleaning_method_final}+sequential_chunk" if cleaning_method_final else 'sequential_chunk'
                    print(f"[JSONWriter] Pipeline cleaning produced {len(cleaned_chunks)} chunks for profile: {profile_id}")
            except Exception as pipeline_error:
                print(f"[JSONWriter] Warning: pipeline cleaning failed: {pipeline_error}. Falling back to text cleaner.")
                try:
                    text_cleaner = get_text_cleaner()
                    cleaned_final = text_cleaner.clean_text(base_text, aggressive=True, use_llm=False)
                    cleaned_chunks = []
                    cleaning_method_final = f"{cleaning_method_final}+regex_fallback" if cleaning_method_final else 'regex_fallback'
                except Exception:
                    cleaned_final = base_text
                    cleaning_method_final = cleaning_method_final or 'raw'

        # Calculate word counts
        word_count_raw = len(combined_text.split()) if combined_text else 0
        word_count_cleaned = len(cleaned_final.split()) if cleaned_final else 0
        
        # Get title (use profile name or first part of profile URL)
        title = profile_name or profile_url.split('/')[-1] or 'Profile'
        
        # Create profile entry
        profile_entry = {
            'id': profile_id,
            'profile_name': profile_name or '',
            'profile_url': profile_url,
            'all_urls': all_urls,
            'title': title,
            'raw_text': combined_text or '',
            'cleaned_text': cleaned_final or '',
            'raw_headings': combined_headings or [],
            'raw_paragraphs': combined_paragraphs or [],
            'chunks': cleaned_chunks,
            'cleaning_status': 'cleaned' if cleaned_final else cleaning_status,
            'cleaning_method': cleaning_method_final,
            'scraped_at': datetime.now().isoformat(),
            'cleaned_at': '',
            'word_count_raw': word_count_raw,
            'word_count_cleaned': word_count_cleaned,
            'chunk_count': len(chunks) if chunks else 0
        }
        
        # Ensure profiles list exists
        if 'profiles' not in self.data:
            self.data['profiles'] = []
        
        # Check if profile already exists (by ID)
        existing_index = None
        for idx, profile in enumerate(self.data['profiles']):
            if profile.get('id') == profile_id:
                existing_index = idx
                break
        
        if existing_index is not None:
            # Update existing profile
            self.data['profiles'][existing_index] = profile_entry
            print(f"[JSONWriter] Updated existing profile - ID: {profile_id}, Name: {profile_name}")
        else:
            # Add new profile
            self.data['profiles'].append(profile_entry)
            print(f"[JSONWriter] Added new profile - ID: {profile_id}, Name: {profile_name}, URLs: {len(all_urls)}, Text length: {len(combined_text)}")
        
        # Update metadata
        if 'metadata' not in self.data:
            self.data['metadata'] = {}
        self.data['metadata']['last_updated'] = datetime.now().isoformat()
        self.data['metadata']['total_profiles'] = len(self.data['profiles'])
    
    def update_cleaned_content(self, profile_id: str, cleaned_text: str, cleaning_method: str = 'regex') -> bool:
        """Update cleaned text for a profile"""
        if 'profiles' not in self.data:
            return False
        
        for profile in self.data['profiles']:
            if profile.get('id') == profile_id:
                profile['cleaned_text'] = cleaned_text
                profile['cleaning_status'] = 'cleaned'
                profile['cleaning_method'] = cleaning_method
                profile['cleaned_at'] = datetime.now().isoformat()
                profile['word_count_cleaned'] = len(cleaned_text.split()) if cleaned_text else 0
                print(f"[JSONWriter] Updated cleaned content for profile ID: {profile_id}")
                return True
        
        print(f"[JSONWriter] Profile ID {profile_id} not found for cleaning update")
        return False
    
    def get_pending_content(self) -> List[Dict]:
        """Get all profiles with pending cleaning status"""
        if 'profiles' not in self.data:
            return []
        
        pending = []
        for profile in self.data['profiles']:
            if profile.get('cleaning_status') == 'pending':
                pending.append({
                    'id': profile.get('id'),
                    'profile_name': profile.get('profile_name'),
                    'raw_text': profile.get('raw_text', ''),
                    'profile_url': profile.get('profile_url', '')
                })
        
        return pending
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """Get a specific profile by ID"""
        if 'profiles' not in self.data:
            return None
        
        for profile in self.data['profiles']:
            if profile.get('id') == profile_id:
                return profile
        
        return None
    
    def get_all_profiles(self) -> List[Dict]:
        """Get all profiles"""
        return self.data.get('profiles', [])
    
    def update_profile_chunks(self, profile_id: str, chunks: List[Dict]) -> bool:
        """
        Update chunks for an existing profile
        
        Args:
            profile_id: Profile ID to update
            chunks: List of chunk dictionaries
        
        Returns:
            True if updated successfully
        """
        if 'profiles' not in self.data:
            return False
        
        for profile in self.data['profiles']:
            if profile.get('id') == profile_id:
                profile['chunks'] = chunks
                profile['chunk_count'] = len(chunks)
                print(f"[JSONWriter] Updated chunks for profile ID: {profile_id} ({len(chunks)} chunks)")
                return True
        
        print(f"[JSONWriter] Profile ID {profile_id} not found for chunk update")
        return False

    def update_profile_name(self, profile_id: str, new_name: str) -> bool:
        """
        Update only the profile_name field.
        """
        if not new_name or 'profiles' not in self.data:
            return False

        for profile in self.data['profiles']:
            if profile.get('id') == profile_id:
                old_name = profile.get('profile_name')
                profile['profile_name'] = new_name
                # keep title in sync if it was identical to the old name
                if profile.get('title') in (old_name, None, ''):
                    profile['title'] = new_name
                self.data.setdefault('metadata', {})['last_updated'] = datetime.now().isoformat()
                print(f"[JSONWriter] Updated profile_name for profile ID: {profile_id} -> {new_name}")
                return True
        return False

    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a profile by ID.
        """
        if 'profiles' not in self.data:
            return False
        original_len = len(self.data['profiles'])
        self.data['profiles'] = [p for p in self.data['profiles'] if p.get('id') != profile_id]
        if len(self.data['profiles']) != original_len:
            self.data.setdefault('metadata', {})['last_updated'] = datetime.now().isoformat()
            print(f"[JSONWriter] Deleted profile ID: {profile_id}")
            return True
        return False
    
    def ensure_all_profiles_have_chunks(self, llm_clean_chunks: bool = True, llm_timeout: int = 12) -> int:
        """
        Ensure all profiles have chunks. If a profile doesn't have chunks,
        create them from the raw_text.
        
        Returns:
            Number of profiles updated
        """
        if 'profiles' not in self.data:
            return 0
        
        updated_count = 0
        
        for profile in self.data['profiles']:
            # Check if profile has chunks
            if not profile.get('chunks') or len(profile.get('chunks', [])) == 0:
                raw_text = profile.get('cleaned_text') or profile.get('raw_text', '')
                if raw_text:
                    try:
                        from api.services.text_chunker import get_text_chunker
                        
                        # Get headings and paragraphs if available
                        headings = profile.get('raw_headings', [])
                        paragraphs = profile.get('raw_paragraphs', [])
                        professor_name = profile.get('profile_name', '')
                        
                        # Create chunks (chunks will have newlines removed automatically)
                        chunker = get_text_chunker(chunk_size=1000, chunk_overlap=None)  # None = 10% overlap
                        chunks = chunker.chunk_structured_text(
                            text=raw_text,
                            headings=headings if headings else None,
                            paragraphs=paragraphs if paragraphs else None,
                            professor_name=professor_name,
                            llm_clean_chunks=llm_clean_chunks,
                            llm_timeout=llm_timeout
                        )
                        
                        # Update profile with chunks
                        profile['chunks'] = chunks
                        profile['chunk_count'] = len(chunks)
                        updated_count += 1
                        print(f"[JSONWriter] Created {len(chunks)} chunks for profile: {profile.get('profile_name', profile.get('id'))}")
                    except Exception as e:
                        print(f"[JSONWriter] Error creating chunks for profile {profile.get('id')}: {str(e)}")
                        # Ensure chunks field exists even if empty
                        if 'chunks' not in profile:
                            profile['chunks'] = []
                            profile['chunk_count'] = 0
        
        if updated_count > 0:
            self.save()
            print(f"[JSONWriter] Updated {updated_count} profiles with chunks")
        
        return updated_count
    
    def save(self):
        """Save data to JSON file"""
        try:
            # Ensure directory exists
            file_dir = os.path.dirname(self.json_file_path)
            if file_dir and not os.path.exists(file_dir):
                os.makedirs(file_dir, exist_ok=True)
            
            # Ensure data structure is valid
            if not isinstance(self.data, dict):
                print(f"[JSONWriter] Warning: data is not a dict, resetting...")
                self.data = {
                    'profiles': [],
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
            
            if 'profiles' not in self.data:
                self.data['profiles'] = []
            
            if 'metadata' not in self.data:
                self.data['metadata'] = {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0'
                }
            
            # Write JSON with pretty formatting
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            print(f"[JSONWriter] ✅ JSON file saved: {self.json_file_path}")
            print(f"[JSONWriter] Total profiles in file: {len(self.data.get('profiles', []))}")
        except Exception as e:
            print(f"[JSONWriter] ❌ ERROR saving JSON file to {self.json_file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Singleton instance
_json_writer = None

def get_json_writer(json_file_path: str = "extracted_content.json") -> JSONWriter:
    """Get or create JSON writer instance"""
    global _json_writer
    
    # Convert to absolute path for comparison (same logic as JSONWriter.__init__)
    if not os.path.isabs(json_file_path):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        api_dir = os.path.dirname(current_file_dir)  # api/services -> api
        project_root = os.path.dirname(api_dir)  # api -> project root
        abs_json_file_path = os.path.join(project_root, json_file_path)
    else:
        abs_json_file_path = json_file_path
    
    # Create new instance if None or if file path changed
    if _json_writer is None or _json_writer.json_file_path != abs_json_file_path:
        _json_writer = JSONWriter(json_file_path)
    
    return _json_writer




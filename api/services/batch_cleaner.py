"""
Batch cleaning service for cleaning raw content stored in JSON
Reads raw content from JSON, cleans it, and updates the cleaned_text field
"""
import os
from typing import List, Dict
from api.services.json_writer import get_json_writer
from api.utils.text_cleaner import get_text_cleaner


class BatchCleaner:
    """Clean raw content stored in JSON in batch"""
    
    def __init__(self, json_file_path: str = "extracted_content.json"):
        self.json_writer = get_json_writer(json_file_path)
        self.text_cleaner = get_text_cleaner()
    
    def clean_all_pending(self, use_llm: bool = True, method: str = 'auto') -> Dict:
        """
        Clean all pending content in JSON
        
        Args:
            use_llm: Whether to use LLM cleaning when cookie content is detected
            method: Cleaning method ('auto', 'regex', 'llm')
                    - 'auto': Use regex first, LLM if cookie content detected
                    - 'regex': Use regex only
                    - 'llm': Use LLM for all
        
        Returns:
            Dictionary with cleaning statistics
        """
        # Get all pending content
        pending_items = self.json_writer.get_pending_content()
        
        if not pending_items:
            print("[BatchCleaner] No pending content to clean")
            return {
                'total': 0,
                'cleaned': 0,
                'failed': 0,
                'skipped': 0
            }
        
        print(f"[BatchCleaner] Found {len(pending_items)} items to clean")
        
        stats = {
            'total': len(pending_items),
            'cleaned': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for i, item in enumerate(pending_items, 1):
            content_id = item['id']
            url = item['url']
            raw_text = item['raw_text']
            
            print(f"[BatchCleaner] [{i}/{len(pending_items)}] Cleaning: {url[:60]}...")
            
            try:
                # Determine cleaning method
                if method == 'llm':
                    cleaned_text = self._clean_with_llm(raw_text)
                    cleaning_method = 'llm'
                elif method == 'regex':
                    cleaned_text = self.text_cleaner.clean_text(raw_text, aggressive=True, use_llm=False)
                    cleaning_method = 'regex'
                else:  # auto
                    # Use regex first, LLM if cookie content detected
                    cleaned_text = self.text_cleaner.clean_text(raw_text, aggressive=True, use_llm=use_llm)
                    cleaning_method = 'regex+llm' if use_llm and self.text_cleaner._has_cookie_content(raw_text) else 'regex'
                
                # Update JSON with cleaned content
                success = self.json_writer.update_cleaned_content(
                    profile_id=content_id,
                    cleaned_text=cleaned_text,
                    cleaning_method=cleaning_method
                )
                
                if success:
                    stats['cleaned'] += 1
                    print(f"[BatchCleaner] ✅ Cleaned: {content_id} ({len(raw_text)} -> {len(cleaned_text)} chars)")
                else:
                    stats['failed'] += 1
                    print(f"[BatchCleaner] ❌ Failed to update JSON for: {content_id}")
                    
            except Exception as e:
                stats['failed'] += 1
                print(f"[BatchCleaner] ❌ Error cleaning {content_id}: {str(e)}")
                # Mark as failed in JSON
                try:
                    self.json_writer.update_cleaned_content(
                        profile_id=content_id,
                        cleaned_text='',
                        cleaning_method='error'
                    )
                    # Manually set status to failed since update_cleaned_content sets it to 'cleaned'
                    for profile in self.json_writer.data.get('profiles', []):
                        if profile.get('id') == content_id:
                            profile['cleaning_status'] = 'failed'
                            break
                except:
                    pass
        
        # Save JSON file
        self.json_writer.save()
        
        print(f"\n[BatchCleaner] ✅ Batch cleaning complete!")
        print(f"  Total: {stats['total']}")
        print(f"  Cleaned: {stats['cleaned']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
        
        return stats
    
    def _clean_with_llm(self, text: str) -> str:
        """Clean text using LLM"""
        try:
            from api.utils.llm_text_cleaner import get_llm_text_cleaner
            llm_cleaner = get_llm_text_cleaner()
            if llm_cleaner:
                return llm_cleaner.clean_text(text, timeout=30, use_chunking=True)
        except Exception as e:
            print(f"[BatchCleaner] LLM cleaning failed: {str(e)}, falling back to regex")
        
        # Fallback to regex
        return self.text_cleaner.clean_text(text, aggressive=True, use_llm=False)
    
    def clean_single_item(self, content_id: str, method: str = 'auto') -> bool:
        """
        Clean a single item by ID
        
        Args:
            content_id: ID of the content to clean
            method: Cleaning method ('auto', 'regex', 'llm')
        
        Returns:
            True if cleaned successfully
        """
        # Get all pending items and find the one with matching ID
        pending_items = self.json_writer.get_pending_content()
        item = next((i for i in pending_items if i['id'] == content_id), None)
        
        if not item:
            print(f"[BatchCleaner] Content ID {content_id} not found or already cleaned")
            return False
        
        raw_text = item['raw_text']
        
        try:
            # Clean the text
            if method == 'llm':
                cleaned_text = self._clean_with_llm(raw_text)
                cleaning_method = 'llm'
            elif method == 'regex':
                cleaned_text = self.text_cleaner.clean_text(raw_text, aggressive=True, use_llm=False)
                cleaning_method = 'regex'
            else:  # auto
                cleaned_text = self.text_cleaner.clean_text(raw_text, aggressive=True, use_llm=True)
                cleaning_method = 'regex+llm' if self.text_cleaner._has_cookie_content(raw_text) else 'regex'
            
            # Update JSON
            success = self.json_writer.update_cleaned_content(
                profile_id=content_id,
                cleaned_text=cleaned_text,
                cleaning_method=cleaning_method
            )
            
            if success:
                self.json_writer.save()
                print(f"[BatchCleaner] ✅ Cleaned: {content_id}")
            
            return success
            
        except Exception as e:
            print(f"[BatchCleaner] ❌ Error cleaning {content_id}: {str(e)}")
            return False


def clean_all_pending_content(json_file_path: str = "extracted_content.json", 
                              use_llm: bool = True, 
                              method: str = 'auto') -> Dict:
    """
    Convenience function to clean all pending content
    
    Args:
        json_file_path: Path to JSON file
        use_llm: Whether to use LLM cleaning
        method: Cleaning method ('auto', 'regex', 'llm')
    
    Returns:
        Dictionary with cleaning statistics
    """
    cleaner = BatchCleaner(json_file_path)
    return cleaner.clean_all_pending(use_llm=use_llm, method=method)


if __name__ == "__main__":
    # Can be run as a script
    import sys
    
    method = 'auto'
    if len(sys.argv) > 1:
        method = sys.argv[1]  # 'auto', 'regex', or 'llm'
    
    print(f"[BatchCleaner] Starting batch cleaning with method: {method}")
    stats = clean_all_pending_content(method=method)
    print(f"\n[BatchCleaner] Done! Stats: {stats}")


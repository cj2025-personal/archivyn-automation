"""
Web scraping service - Controller Layer (Windows Safe Version)
Delegates ALL extraction logic to 'playwright_subprocess_worker.py'.
Uses subprocess.run inside a thread to avoid Windows Asyncio Event Loop crashes.
"""

import asyncio
import sys
import os
import json
import subprocess
from urllib.parse import urlparse
from typing import Dict

class ProfileScraper:
    def __init__(self):
        # 1. Resolve Worker Path Robustly
        # Assuming structure: project_root/api/services/scraper.py
        # We need to go up 2 levels to get to project_root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        self._worker_path = os.path.join(project_root, 'playwright_subprocess_worker.py')
        
        # Fallback: Try looking in the same directory if the relative path failed
        if not os.path.exists(self._worker_path):
            self._worker_path = os.path.join(current_dir, 'playwright_subprocess_worker.py')
        
        print(f"📍 [Controller] Configured worker path: {self._worker_path}")
        
        if not os.path.exists(self._worker_path):
            print(f"❌ [Controller] CRITICAL: Worker script missing!")
            raise FileNotFoundError(f"Worker not found at: {self._worker_path}")

    async def _run_worker(self, task_payload: Dict, timeout: int = 240) -> Dict:
        """
        Runs the worker using subprocess.run (Synchronous but Threaded).
        This is much more stable on Windows than create_subprocess_exec.
        """
        print(f"🚀 [Controller] Spawning worker for: {task_payload.get('url')}")
        
        try:
            # Force UTF-8 handling for Windows to prevent encoding crashes
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            # Define the blocking task
            def run_sync_process():
                return subprocess.run(
                    [sys.executable, self._worker_path],
                    input=json.dumps(task_payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                    timeout=timeout,
                encoding='utf-8',
                    errors='replace',
                env=env
            )
            
            # Run in a thread so we don't block the web server
            process = await asyncio.to_thread(run_sync_process)

            # --- 1. PRINT LOGS (Standard Error) ---
            if process.stderr:
                print("\n" + "="*40)
                print("      🕵️  WORKER LOGS")
                print("="*40)
                # Clean up logs for display
                logs = process.stderr.strip()
                for line in logs.splitlines():
                    if "[Worker]" in line:
                        print(f"  > {line.replace('[Worker]', '').strip()}")
                    else:
                        print(f"    {line}")
                print("="*40 + "\n")

            # --- 2. CHECK FOR CRASHES ---
            if process.returncode != 0:
                print(f"❌ [Controller] Worker crashed (Code {process.returncode})")
                # Try to extract python tracebacks from stderr
                stderr_text = process.stderr or ""
                error_msg = stderr_text[-200:] if len(stderr_text) > 200 else stderr_text
                return {"status": "error", "error": f"Worker failed. logs: {error_msg}"}

            # --- 3. PARSE JSON (Standard Output) ---
            output_str = process.stdout.strip() if process.stdout else ""
            json_result = None
            
            # Scan backwards for JSON (ignores any print statements that might have sneaked into stdout)
            if output_str:
                for line in reversed(output_str.split('\n')):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            json_result = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
            
            if not json_result:
                output_preview = output_str[:200] if output_str else "(empty)"
                print(f"❌ [Controller] No JSON in output. Raw stdout: {output_preview}...")
                return {"status": "error", "error": "Invalid JSON output"}

            return json_result

        except subprocess.TimeoutExpired:
            print(f"⏰ [Controller] Worker timed out after {timeout}s")
            return {"status": "error", "error": "Worker timeout"}
            
        except Exception as e:
            import traceback
            print(f"💥 [Controller] Exception in _run_worker: {e}")
            print(traceback.format_exc())
            return {"status": "error", "error": str(e)}

    async def extract_all(self, profile_url: str) -> Dict:
        """Main entry point."""
        # 1. Run Worker
        worker_result = await self._run_worker({
            'task': 'extract',
            'url': profile_url
        })

        # [FIX] Defensive check for None
        if not worker_result:
            return {
                'profile_data': {'profile_url': profile_url, 'error': 'Worker returned no result'},
                'all_urls': [],
                'document_links': [],
                'text_content': {'full_text': ''},
                'extraction_metadata': {'status': 'failed', 'error': 'Worker returned no result'}
            }

        if worker_result.get('status') != 'success':
            return {
                'profile_data': {'profile_url': profile_url, 'error': worker_result.get('error')},
                'all_urls': [],
                'document_links': [],
                'text_content': {'full_text': ''},
                'extraction_metadata': {'status': 'failed', 'error': worker_result.get('error')}
            }

        print("✅ [Controller] Worker success. Formatting data...")

        # 2. Map Data (Same as before)
        # [FIX] Defensive checks for None values
        raw_profile = worker_result.get('profile_data') or {}
        cv_docs = worker_result.get('cv_documents') or []
        websites = worker_result.get('personal_websites') or []
        profile_page = worker_result.get('profile_page') or {}

        profile_data = {
            'name': raw_profile.get('name', ''),
            'university': urlparse(profile_url).netloc.replace('www.', ''),
            'department': raw_profile.get('department', ''),
            'position': raw_profile.get('position', ''),
            'email': raw_profile.get('email', ''),
            'profile_url': profile_url,
            'full_text': raw_profile.get('full_text', ''),
            'bio': '', 
            'publications': '', 
            'research_interests': ''
        }
        
        # [FIX] Clean name - remove school/university names
        if profile_data['name']:
            import re
            # Remove common school/university patterns
            name = profile_data['name']
            # Remove patterns like "School of X" or "University of X"
            name = re.sub(r'\b(School\s+of|University\s+of|College\s+of).*$', '', name, flags=re.I)
            # Remove if it contains "Indiana University" or similar
            if any(x in name for x in ['Indiana University', 'School of', 'University', 'College']):
                # Try to extract just the person's name (usually first part before school name)
                parts = re.split(r'\s+(School|University|College)', name, flags=re.I)
                if parts and len(parts) > 0 and len(parts[0]) > 2 and len(parts[0]) < 100:
                    name = parts[0].strip()
                else:
                    name = ''  # Clear if it's clearly not a person's name
            profile_data['name'] = name.strip()
        
        # [FIX] Clean email - remove any trailing non-email text
        if profile_data['email']:
            import re
            # Extract only the email part (remove trailing words like "Research", "Email", etc.)
            email_match = re.match(r'^([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', profile_data['email'])
            if email_match:
                profile_data['email'] = email_match.group(1)
            else:
                # If regex doesn't match, try to find email in the string
                emails = re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', profile_data['email'])
                if emails:
                    profile_data['email'] = emails[0]
                else:
                    profile_data['email'] = ''  # Clear if invalid
        
        # [FIX] Clean position - remove navigation menu text
        if profile_data['position']:
            import re
            # If position is too long or contains navigation keywords, try to extract just the position
            if len(profile_data['position']) > 200 or any(x in profile_data['position'].lower() for x in ['undergraduate', 'graduate', 'masters', 'menu', 'navigation']):
                # Try to find position keywords
                position_match = re.search(r'\b(Professor|Associate\s+Professor|Assistant\s+Professor|Lecturer|Director|Chair|Dean|Instructor)\b', profile_data['position'], re.I)
                if position_match:
                    profile_data['position'] = position_match.group(1)
                else:
                    # If no match, try to get first reasonable sentence
                    sentences = re.split(r'[.!?]\s+', profile_data['position'])
                    if sentences:
                        for sent in sentences:
                            if sent and len(sent) < 100 and any(x in sent.lower() for x in ['professor', 'assistant', 'associate', 'lecturer', 'director']):
                                profile_data['position'] = sent.strip()
                                break
                    # If still too long, clear it
                    if len(profile_data['position']) > 200:
                        profile_data['position'] = ''

        all_urls = []
        body_links_with_content = []
        source_records = []

        # Profile page as a source record (if available)
        if profile_page.get("content"):
            source_records.append({
                "source_type": "profile_page",
                "source_url": profile_page.get("source_url") or profile_url,
                "resolved_url": profile_page.get("resolved_url") or profile_url,
                "title": profile_page.get("page_title", ""),
                "content": profile_page.get("content", ""),
                "processing_status": "success",
                "fetch_metadata": profile_page.get("fetch_metadata"),
            })

        for cv in cv_docs:
            # [FIX] Defensive check - ensure cv is a dict
            if not isinstance(cv, dict):
                        continue
                    
            cv_entry = {
                'url': cv.get('url', ''),
                'link_text': 'CV / Resume',
                'type': 'external',
                'category': 'cv',
                'is_document': True,
                'file_type': 'pdf' if 'pdf' in cv.get('type', '') else 'doc',
                'resolved_url': cv.get('resolved_url', ''),
            }
            all_urls.append(cv_entry)
            body_links_with_content.append({
                **cv_entry,
                'content': cv.get('content', ''),
                'content_length': cv.get('full_length', 0),
                'processing_status': cv.get('status', 'unknown'),
                'note': cv.get('note', ''),
                'fetch_metadata': cv.get('fetch_metadata'),
            })
            # Source record for CV
            source_records.append({
                "source_type": "cv",
                "source_url": cv.get("url", ""),
                "resolved_url": cv.get("resolved_url", ""),
                "title": "CV / Resume",
                "content": cv.get("content", ""),
                "processing_status": cv.get("status", "unknown"),
                "fetch_metadata": cv.get("fetch_metadata"),
                "note": cv.get("note", ""),
            })

        for site in websites:
            # [FIX] Defensive check - ensure site is a dict
            if not isinstance(site, dict):
                    continue
            
            site_entry = {
                'url': site.get('url', ''),
                'link_text': 'Personal Website',
                'type': 'external',
                'category': 'personal_website',
                'is_document': False,
                'resolved_url': site.get('resolved_url', site.get('url', '')),
            }
            all_urls.append(site_entry)
            body_links_with_content.append({
                **site_entry,
                'content': site.get('content', ''),
                'processing_status': 'success',
                'fetch_metadata': site.get('fetch_metadata'),
            })
            source_records.append({
                "source_type": "personal_website",
                "source_url": site.get("source_url") or site.get("url", ""),
                "resolved_url": site.get("resolved_url") or site.get("url", ""),
                "title": "Personal Website",
                "content": site.get("content", ""),
                "processing_status": site.get("status", "success"),
                "fetch_metadata": site.get("fetch_metadata"),
            })
            
            for sub in site.get('subpages', []):
                # [FIX] Defensive check for subpages
                if not isinstance(sub, dict):
                        continue
                    
                sub_entry = {
                    'url': sub.get('url', ''),
                    'link_text': 'Website Subpage',
                    'type': 'external',
                    'category': 'personal_website_subpage',
                    'is_document': False,
                    'resolved_url': sub.get('resolved_url', sub.get('url', '')),
                }
                all_urls.append(sub_entry)
                body_links_with_content.append({
                    **sub_entry,
                    'content': sub.get('content', ''),
                    'processing_status': 'success',
                    'fetch_metadata': sub.get('fetch_metadata'),
                })
                source_records.append({
                    "source_type": "personal_website_subpage",
                    "source_url": sub.get("source_url") or sub.get("url", ""),
                    "resolved_url": sub.get("resolved_url") or sub.get("url", ""),
                    "title": "Website Subpage",
                    "content": sub.get("content", ""),
                    "processing_status": sub.get("status", "success"),
                    "fetch_metadata": sub.get("fetch_metadata"),
                })

        # Always return a structured response even if no websites/cvs were found
        return {
            'profile_data': profile_data,
            'all_urls': all_urls,
            'document_links': [u for u in all_urls if u and isinstance(u, dict) and u.get('is_document')],
            'text_content': {'full_text': profile_data['full_text']},
            'body_links_with_content': body_links_with_content,
            'source_records': source_records,
            'extraction_metadata': {
                'source': 'subprocess_worker',
                'cv_count': len(cv_docs),
                'website_count': len(websites),
                'total_subpages': (worker_result.get('summary') or {}).get('total_subpages', 0),
                'profile_url': profile_url
            }
        }

    async def scrape_webpage(self, url: str) -> Dict:
        """Scrape a single generic webpage."""
        result = await self._run_worker({'task': 'extract', 'url': url})
        if result.get('status') != 'success':
            raise Exception(f"Worker failed: {result.get('error')}")
        data = result.get('profile_data', {})
        return {
            'url': url,
            'title': data.get('name', ''),
            'text_content': {'full_text': data.get('full_text', '')},
            'html_content': ''
        }

    async def close_browser(self):
        pass

# Singleton instance
_scraper_instance = None

async def get_scraper() -> ProfileScraper:
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = ProfileScraper()
    return _scraper_instance

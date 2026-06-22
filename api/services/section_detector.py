"""
Section detection service
Detects sections in text using pattern matching or LLM
Respects LLM context window limits
"""
import re
from typing import List, Dict, Optional, Tuple
import os


class SectionDetector:
    """Detect sections in text using pattern matching or LLM"""
    
    def __init__(
        self,
        use_llm: bool = False,
        ollama_model: str = "llama3",
        ollama_host: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        max_context_tokens: int = 4000  # Leave buffer for response
    ):
        """
        Initialize section detector
        
        Args:
            use_llm: Whether to use LLM for section detection
            ollama_model: Ollama model name
            ollama_host: Ollama host URL
            openai_model: OpenAI model name
            max_context_tokens: Maximum tokens for LLM context (default 4000, leaving buffer)
        """
        self.use_llm = use_llm
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY')
        self.openai_model = openai_model
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.max_context_tokens = max_context_tokens
        
        # Model context window sizes (approximate)
        self.model_context_windows = {
            'llama3': 8192,
            'llama3.1': 128000,
            'mistral': 8192,
            'gpt-3.5-turbo': 16385,
            'gpt-4': 8192,
            'gpt-4-turbo': 128000,
        }
    
    def detect_sections(self, text: str, use_llm: Optional[bool] = None) -> List[Dict]:
        """
        Detect sections in text
        
        Args:
            text: Text to analyze
            use_llm: Override default LLM usage
        
        Returns:
            List of section dictionaries with 'start', 'end', 'title', 'type' keys
        """
        if use_llm is None:
            use_llm = self.use_llm
        
        # Prioritize LLM for section detection if enabled
        if use_llm:
            try:
                llm_sections = self._detect_sections_llm(text)
                if llm_sections and len(llm_sections) > 0:
                    print(f"[SectionDetector] Using LLM-detected sections: {len(llm_sections)} sections found")
                    return llm_sections
            except Exception as e:
                print(f"[SectionDetector] LLM detection failed: {str(e)}, falling back to pattern-based detection")
                import traceback
                traceback.print_exc()
        
        # Fallback to pattern-based detection
        sections = self._detect_sections_pattern(text)
        return sections
    
    def _detect_sections_pattern(self, text: str) -> List[Dict]:
        """
        Detect sections using pattern matching
        
        Args:
            text: Text to analyze
        
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Pattern 1: Section markers like "=== SECTION ==="
        pattern1 = r'===+\s*([^=]+?)\s*===+'
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            section_start = match.start()
            section_title = match.group(1).strip()
            
            # Find end of section (next section marker or end of text)
            section_end = len(text)
            next_match = re.search(pattern1, text[section_start + len(match.group(0)):], re.IGNORECASE)
            if next_match:
                section_end = section_start + len(match.group(0)) + next_match.start()
            
            sections.append({
                'start': section_start,
                'end': section_end,
                'title': section_title,
                'type': self._classify_section_type(section_title),
                'method': 'pattern'
            })
        
        # Pattern 2: Markdown-style headings (# ## ###)
        if not sections:
            pattern2 = r'^(#{1,6})\s+(.+?)$'
            for match in re.finditer(pattern2, text, re.MULTILINE):
                section_start = match.start()
                section_title = match.group(2).strip()
                heading_level = len(match.group(1))
                
                # Find end of section (next heading of same or higher level, or end)
                section_end = len(text)
                next_pattern = r'^#{1,' + str(heading_level) + r'}\s+'
                next_match = re.search(next_pattern, text[section_start + len(match.group(0)):], re.MULTILINE)
                if next_match:
                    section_end = section_start + len(match.group(0)) + next_match.start()
                
                sections.append({
                    'start': section_start,
                    'end': section_end,
                    'title': section_title,
                    'type': self._classify_section_type(section_title),
                    'method': 'pattern'
                })
        
        # Pattern 3: ALL CAPS headings (common in CVs/resumes)
        if not sections:
            pattern3 = r'^([A-Z][A-Z\s]{5,}):?\s*$'
            for match in re.finditer(pattern3, text, re.MULTILINE):
                section_start = match.start()
                section_title = match.group(1).strip()
                
                # Find end of section
                section_end = len(text)
                next_match = re.search(pattern3, text[section_start + len(match.group(0)):], re.MULTILINE)
                if next_match:
                    section_end = section_start + len(match.group(0)) + next_match.start()
                
                sections.append({
                    'start': section_start,
                    'end': section_end,
                    'title': section_title,
                    'type': self._classify_section_type(section_title),
                    'method': 'pattern'
                })
        
        # Pattern 4: Common section keywords (including variations like "PUBLICATIONS AND PRESENTATIONS:")
        if not sections:
            section_keywords = [
                r'EDUCATION',
                r'EXPERIENCE',
                r'PUBLICATIONS\s+AND\s+PRESENTATIONS',
                r'PUBLICATIONS',
                r'PRESENTATIONS',
                r'RESEARCH',
                r'AWARDS',
                r'HONORS',
                r'PROFESSIONAL',
                r'CONTACT',
                r'BIOGRAPHY',
                r'ABOUT',
                r'SKILLS',
                r'PROJECTS',
                r'TEACHING',
                r'APPOINTMENTS',
            ]
            
            for keyword_pattern in section_keywords:
                # Handle patterns with spaces (like "PUBLICATIONS AND PRESENTATIONS")
                pattern = rf'{keyword_pattern}[\s:]*'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    section_start = match.start()
                    section_title = match.group(0).strip().rstrip(':').strip()
                    
                    # Find end (next keyword or end)
                    section_end = len(text)
                    for next_keyword in section_keywords:
                        if next_keyword != keyword_pattern:
                            next_pattern = rf'{next_keyword}[\s:]*'
                            next_match = re.search(next_pattern, text[section_start + len(match.group(0)):], re.IGNORECASE)
                            if next_match and section_start + len(match.group(0)) + next_match.start() < section_end:
                                section_end = section_start + len(match.group(0)) + next_match.start()
                    
                    sections.append({
                        'start': section_start,
                        'end': section_end,
                        'title': section_title,
                        'type': self._classify_section_type(section_title),
                        'method': 'pattern'
                    })
                    break  # Only take first match per keyword
        
        # Sort sections by start position
        sections.sort(key=lambda x: x['start'])
        
        # Merge overlapping sections
        sections = self._merge_overlapping_sections(sections)
        
        return sections
    
    def _detect_sections_llm(self, text: str) -> List[Dict]:
        """
        Detect sections using LLM
        
        Args:
            text: Text to analyze
        
        Returns:
            List of section dictionaries
        """
        # Check if text is too long for LLM context window
        text_length = len(text)
        estimated_tokens = text_length // 4  # Rough estimate: 1 token ≈ 4 chars
        
        # Get model context window
        model_name = self.ollama_model if self.ollama_api_key or not self.openai_api_key else self.openai_model
        context_window = self.model_context_windows.get(model_name, 4000)
        
        # If text is too long, chunk it and process each chunk
        if estimated_tokens > self.max_context_tokens:
            print(f"[SectionDetector] Text too long ({estimated_tokens} est. tokens), chunking for LLM analysis")
            return self._detect_sections_llm_chunked(text, context_window)
        
        # Try OpenAI first if available
        if self.openai_api_key:
            try:
                return self._detect_sections_openai(text)
            except Exception as e:
                print(f"[SectionDetector] OpenAI detection failed: {str(e)}")
        
        # Try Ollama
        if self.ollama_api_key or not self.openai_api_key:
            try:
                return self._detect_sections_ollama(text)
            except Exception as e:
                print(f"[SectionDetector] Ollama detection failed: {str(e)}")
        
        # Fallback to pattern-based
        return self._detect_sections_pattern(text)
    
    def _detect_sections_llm_chunked(self, text: str, context_window: int) -> List[Dict]:
        """
        Detect sections in long text by chunking and processing with LLM
        
        Args:
            text: Long text to analyze
            context_window: Model context window size
        
        Returns:
            List of section dictionaries
        """
        # Chunk text with overlap
        chunk_size = self.max_context_tokens * 3  # Characters per chunk (rough estimate)
        overlap = chunk_size // 4
        
        all_sections = []
        start_pos = 0
        
        while start_pos < len(text):
            chunk_end = min(start_pos + chunk_size, len(text))
            chunk = text[start_pos:chunk_end]
            
            # Adjust start_pos for overlap on next iteration
            if chunk_end < len(text):
                # Find a good break point (sentence boundary)
                for i in range(chunk_end - 1, max(start_pos, chunk_end - overlap), -1):
                    if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                        chunk_end = i + 1
                        break
            
            chunk_text = text[start_pos:chunk_end]
            
            # Detect sections in this chunk
            try:
                if self.openai_api_key:
                    chunk_sections = self._detect_sections_openai(chunk_text)
                else:
                    chunk_sections = self._detect_sections_ollama(chunk_text)
                
                # Adjust section positions to account for chunk offset
                for section in chunk_sections:
                    section['start'] += start_pos
                    section['end'] += start_pos
                
                all_sections.extend(chunk_sections)
            except Exception as e:
                print(f"[SectionDetector] Error detecting sections in chunk: {str(e)}")
            
            # Move to next chunk with overlap
            start_pos = chunk_end - overlap if chunk_end < len(text) else len(text)
        
        # Merge overlapping sections and remove duplicates
        all_sections.sort(key=lambda x: x['start'])
        return self._merge_overlapping_sections(all_sections)
    
    def _detect_sections_openai(self, text: str) -> List[Dict]:
        """Detect sections using OpenAI"""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.openai_api_key)
        
        # Limit text size to fit in context window
        max_chars = self.max_context_tokens * 3  # Rough estimate: 1 token ≈ 4 chars, leave buffer
        text_to_analyze = text[:max_chars] if len(text) > max_chars else text
        
        prompt = f"""Analyze the following text and identify all major sections. 
For each section, provide:
- section_title: The title/heading of the section
- start_position: Character position where section starts (approximate, relative to the provided text)
- section_type: Type of section (e.g., "education", "experience", "publications", "research", "awards", "contact", "biography", "other")

Text to analyze:
{text_to_analyze}

Return a JSON array of sections in this format:
[
  {{
    "section_title": "Education",
    "start_position": 0,
    "section_type": "education"
  }},
  ...
]

Only return the JSON array, no other text."""

        try:
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a text analysis expert. Identify sections in text and return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                import json
                sections_data = json.loads(json_match.group(0))
                
                # Convert to our format
                sections = []
                for sec in sections_data:
                    # Find actual position in text
                    title = sec.get('section_title', '')
                    start_pos = self._find_section_position(text, title, sec.get('start_position', 0))
                    
                    sections.append({
                        'start': start_pos,
                        'end': len(text),  # Will be adjusted by merge
                        'title': title,
                        'type': sec.get('section_type', 'other'),
                        'method': 'llm_openai'
                    })
                
                # Adjust end positions
                sections.sort(key=lambda x: x['start'])
                for i in range(len(sections) - 1):
                    sections[i]['end'] = sections[i + 1]['start']
                
                return sections
        except Exception as e:
            print(f"[SectionDetector] OpenAI API error: {str(e)}")
            raise
        
        return []
    
    def _detect_sections_ollama(self, text: str) -> List[Dict]:
        """Detect sections using Ollama"""
        try:
            import ollama
            import json
        except ImportError:
            raise ImportError("ollama package not installed")
        
        # Limit text size to fit in context window
        max_chars = self.max_context_tokens * 3  # Rough estimate: 1 token ≈ 4 chars, leave buffer
        text_to_analyze = text[:max_chars] if len(text) > max_chars else text
        
        prompt = f"""Analyze the following text and identify all major sections. 
For each section, provide:
- section_title: The title/heading of the section
- start_position: Character position where section starts (approximate, relative to the provided text)
- section_type: Type of section (e.g., "education", "experience", "publications", "research", "awards", "contact", "biography", "other")

Text to analyze:
{text_to_analyze}

Return a JSON array of sections in this format:
[
  {{
    "section_title": "Education",
    "start_position": 0,
    "section_type": "education"
  }},
  ...
]

Only return the JSON array, no other text."""

        try:
            if self.ollama_api_key:
                # Use Ollama Cloud API
                import requests
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    },
                    headers={"Authorization": f"Bearer {self.ollama_api_key}"},
                    timeout=60
                )
                result_text = response.json().get('response', '')
            else:
                # Use local Ollama
                response = ollama.generate(
                    model=self.ollama_model,
                    prompt=prompt
                )
                result_text = response['response']
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                sections_data = json.loads(json_match.group(0))
                
                # Convert to our format
                sections = []
                for sec in sections_data:
                    title = sec.get('section_title', '')
                    start_pos = self._find_section_position(text, title, sec.get('start_position', 0))
                    
                    sections.append({
                        'start': start_pos,
                        'end': len(text),
                        'title': title,
                        'type': sec.get('section_type', 'other'),
                        'method': 'llm_ollama'
                    })
                
                # Adjust end positions
                sections.sort(key=lambda x: x['start'])
                for i in range(len(sections) - 1):
                    sections[i]['end'] = sections[i + 1]['start']
                
                return sections
        except Exception as e:
            print(f"[SectionDetector] Ollama API error: {str(e)}")
            raise
        
        return []
    
    def _find_section_position(self, text: str, title: str, estimated_pos: int) -> int:
        """Find actual position of section title in text"""
        # Try exact match first
        pos = text.find(title, max(0, estimated_pos - 100))
        if pos >= 0:
            return pos
        
        # Try case-insensitive match
        pos = text.lower().find(title.lower(), max(0, estimated_pos - 100))
        if pos >= 0:
            return pos
        
        # Try partial match
        words = title.split()
        if words:
            first_word = words[0]
            pos = text.lower().find(first_word.lower(), max(0, estimated_pos - 200))
            if pos >= 0:
                return pos
        
        return estimated_pos
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()
        
        if any(kw in title_lower for kw in ['education', 'degree', 'university', 'school', 'academic']):
            return 'education'
        elif any(kw in title_lower for kw in ['experience', 'employment', 'work', 'position', 'career', 'appointment']):
            return 'experience'
        elif any(kw in title_lower for kw in ['publication', 'paper', 'article', 'journal', 'conference']):
            return 'publications'
        elif any(kw in title_lower for kw in ['research', 'interest', 'expertise', 'area']):
            return 'research'
        elif any(kw in title_lower for kw in ['award', 'honor', 'recognition', 'achievement']):
            return 'awards'
        elif any(kw in title_lower for kw in ['contact', 'email', 'phone', 'address']):
            return 'contact'
        elif any(kw in title_lower for kw in ['bio', 'biography', 'about', 'profile', 'background']):
            return 'biography'
        elif any(kw in title_lower for kw in ['skill', 'competence', 'ability']):
            return 'skills'
        elif any(kw in title_lower for kw in ['project', 'work']):
            return 'projects'
        else:
            return 'other'
    
    def _merge_overlapping_sections(self, sections: List[Dict]) -> List[Dict]:
        """Merge overlapping sections, keeping the most specific one"""
        if not sections:
            return []
        
        # Sort by start position
        sections.sort(key=lambda x: x['start'])
        
        merged = []
        current = sections[0]
        
        for next_section in sections[1:]:
            # If sections overlap
            if current['end'] > next_section['start']:
                # Keep the one with more specific detection method or longer title
                if (next_section.get('method', '') == 'llm' and current.get('method', '') != 'llm') or \
                   len(next_section['title']) > len(current['title']):
                    current = next_section
                # Extend current section's end
                current['end'] = max(current['end'], next_section['end'])
            else:
                merged.append(current)
                current = next_section
        
        merged.append(current)
        return merged


# Singleton instance
_section_detector = None

def get_section_detector(
    use_llm: bool = False,
    ollama_model: str = "llama3",
    ollama_host: Optional[str] = None,
    openai_model: str = "gpt-3.5-turbo",
    max_context_tokens: int = 4000
) -> SectionDetector:
    """Get or create section detector instance"""
    global _section_detector
    
    if _section_detector is None:
        _section_detector = SectionDetector(
            use_llm=use_llm,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            openai_model=openai_model,
            max_context_tokens=max_context_tokens
        )
    
    return _section_detector


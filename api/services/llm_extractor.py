"""
LLM-based structured data extraction service
Uses Ollama (local/remote/cloud) or can be configured for other LLM APIs
"""
from typing import Dict, Optional
import json
import re
import os


class LLMExtractor:
    """Extract structured data from unstructured text using LLM"""
    
    def __init__(
        self, 
        use_ollama: bool = True, 
        ollama_model: str = "llama3",
        ollama_host: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo"
    ):
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        # Support remote Ollama: get from parameter, env var, or default to localhost
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        # Support Ollama Cloud API
        self.ollama_api_key = ollama_api_key or os.getenv('OLLAMA_API_KEY')
        # Support OpenAI as alternative
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_model = openai_model
    
    def extract_from_text(self, text: str, source_type: str = "cv") -> Dict:
        """
        Extract structured data from text
        
        Args:
            text: Raw text content
            source_type: Type of source (cv, website, profile)
        
        Returns:
            Structured data dictionary
        """
        if not text or len(text.strip()) < 50:
            return self._get_empty_structure()
        
        # Try LLM extraction first
        if self.use_openai and self.openai_api_key:
            try:
                return self._extract_with_openai(text, source_type)
            except Exception as e:
                print(f"OpenAI extraction failed: {str(e)}, falling back to pattern matching")
        elif self.use_ollama:
            try:
                return self._extract_with_ollama(text, source_type)
            except Exception as e:
                print(f"Ollama extraction failed: {str(e)}, falling back to pattern matching")
        
        # Fallback to pattern matching
        return self._extract_with_patterns(text)
    
    def _extract_with_ollama(self, text: str, source_type: str) -> Dict:
        """Extract using Ollama LLM (local, remote, or cloud)"""
        try:
            import httpx
            
            prompt = self._create_extraction_prompt(text, source_type)
            
            # Determine API endpoint
            if self.ollama_api_key:
                # Using Ollama Cloud API
                api_url = 'https://api.ollama.com/v1/chat/completions'
                headers = {
                    'Authorization': f'Bearer {self.ollama_api_key}',
                    'Content-Type': 'application/json'
                }
                payload = {
                    'model': self.ollama_model,
                    'messages': [
                        {'role': 'user', 'content': prompt}
                    ],
                    'stream': False
                }
            else:
                # Using local or remote Ollama server
                api_url = f'{self.ollama_host.rstrip("/")}/api/generate'
                headers = {'Content-Type': 'application/json'}
                payload = {
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                }
            
            # Make API request
            with httpx.Client(timeout=120.0) as http_client:
                resp = http_client.post(
                    api_url,
                    json=payload,
                    headers=headers
                )
                resp.raise_for_status()
                result = resp.json()
                
                # Handle different response formats
                if self.ollama_api_key:
                    # Ollama Cloud format
                    response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    # Standard Ollama format
                    response_text = result.get('response', '')
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Fallback to pattern matching
                return self._extract_with_patterns(text)
        
        except ImportError:
            print("httpx not available, using pattern matching")
            return self._extract_with_patterns(text)
        except Exception as e:
            print(f"Error with Ollama extraction: {str(e)}")
            return self._extract_with_patterns(text)
    
    def _extract_with_openai(self, text: str, source_type: str) -> Dict:
        """Extract using OpenAI API"""
        try:
            import httpx
            
            prompt = self._create_extraction_prompt(text, source_type)
            
            api_url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': self.openai_model,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'response_format': {'type': 'json_object'}
            }
            
            with httpx.Client(timeout=120.0) as http_client:
                resp = http_client.post(
                    api_url,
                    json=payload,
                    headers=headers
                )
                resp.raise_for_status()
                result = resp.json()
                response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Parse JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return self._extract_with_patterns(text)
        
        except ImportError:
            print("httpx not available, using pattern matching")
            return self._extract_with_patterns(text)
        except Exception as e:
            print(f"Error with OpenAI extraction: {str(e)}")
            return self._extract_with_patterns(text)
    
    def _create_extraction_prompt(self, text: str, source_type: str) -> str:
        """Create prompt for LLM extraction"""
        prompt = f"""Extract structured information from the following {source_type} text. 
Return the information as a JSON object with the following structure:

{{
  "bio": "Full biographical information and background",
  "education": [
    {{
      "degree": "Degree name",
      "field": "Field of study",
      "institution": "Institution name",
      "year": "Year completed"
    }}
  ],
  "publications": [
    {{
      "title": "Publication title",
      "authors": "Author names",
      "year": "Year",
      "type": "Journal Article/Book/Conference Paper/etc.",
      "journal": "Journal or publisher name"
    }}
  ],
  "awards": [
    {{
      "name": "Award name",
      "year": "Year",
      "organization": "Organization"
    }}
  ],
  "expertise": ["Research area 1", "Research area 2"],
  "experience": [
    {{
      "position": "Job title",
      "institution": "Institution",
      "start_year": "Start year",
      "end_year": "End year or 'Current'"
    }}
  ],
  "milestones": ["Major achievement 1", "Major achievement 2"]
}}

Text to extract from:
{text[:5000]}

Return only valid JSON, no additional text:"""
        
        return prompt
    
    def _extract_with_patterns(self, text: str) -> Dict:
        """Fallback pattern-based extraction"""
        result = self._get_empty_structure()
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            result['contact_info'] = emails[0]
        
        # Extract years (for education, awards, etc.)
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        
        # Try to extract education (look for degree keywords)
        degree_keywords = ['phd', 'ph.d', 'doctorate', 'master', 'bachelor', 'b.s', 'm.s', 'm.a', 'b.a']
        education_section = self._extract_section(text, degree_keywords)
        if education_section:
            result['education'] = [{'description': education_section}]
        
        # Extract publications (look for common patterns)
        # This is simplified - LLM would do better
        publication_keywords = ['publication', 'journal', 'conference', 'paper', 'article']
        pub_section = self._extract_section(text, publication_keywords)
        
        # Extract expertise/research areas
        # Look for section headers or keywords
        expertise_keywords = ['research', 'expertise', 'interest', 'focus', 'area']
        expertise_section = self._extract_section(text, expertise_keywords)
        if expertise_section:
            # Simple keyword extraction
            result['expertise'] = self._extract_keywords(expertise_section)
        
        # Bio is the full text if no specific section found
        result['bio'] = text[:1000] if len(text) > 1000 else text
        
        return result
    
    def _extract_section(self, text: str, keywords: list) -> Optional[str]:
        """Extract a section that contains keywords"""
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                in_section = True
                section_lines.append(line)
            elif in_section:
                # Continue until empty line or new section
                if line.strip():
                    section_lines.append(line)
                else:
                    break
        
        return '\n'.join(section_lines) if section_lines else None
    
    def _extract_keywords(self, text: str) -> list:
        """Extract potential research areas/keywords"""
        # Simple keyword extraction - can be improved
        # Look for capitalized phrases, technical terms, etc.
        words = text.split()
        keywords = []
        
        # Look for capitalized phrases (potential research areas)
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 3:
                # Check if it's part of a phrase
                if i < len(words) - 1 and words[i+1] and words[i+1][0].isupper():
                    phrase = f"{word} {words[i+1]}"
                    if phrase not in keywords:
                        keywords.append(phrase)
        
        return keywords[:10]  # Limit to 10
    
    def _get_empty_structure(self) -> Dict:
        """Return empty structured data structure"""
        return {
            'bio': '',
            'education': [],
            'publications': [],
            'awards': [],
            'expertise': [],
            'experience': [],
            'milestones': []
        }


# Singleton instance
_llm_extractor = None

def get_llm_extractor(
    use_ollama: bool = True, 
    ollama_model: str = "llama3",
    ollama_host: Optional[str] = None,
    ollama_api_key: Optional[str] = None,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-3.5-turbo"
) -> LLMExtractor:
    """Get or create LLM extractor instance"""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMExtractor(
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_api_key=ollama_api_key,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_model=openai_model
        )
    return _llm_extractor


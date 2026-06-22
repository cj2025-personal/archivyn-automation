"""
Text cleaning utility for removing noise, boilerplate, and irrelevant content
from scraped web content and documents.
"""
import re
from typing import Dict, List, Optional
import html


class TextCleaner:
    """Clean and normalize extracted text content"""
    
    def __init__(self):
        # Common cookie/privacy notice patterns (without (?i) flag - will be added during compilation)
        self.cookie_patterns = [
            r'cookie\s*(?:policy|notice|consent|settings|preferences|information|banner|popup|dialog)',
            r'accept\s*(?:all\s*)?cookies?',
            r'reject\s*(?:all\s*)?cookies?',
            r'we\s+use\s+cookies?',
            r'this\s+website\s+uses\s+cookies?',
            r'by\s+continuing\s+to\s+(?:browse|use)\s+(?:without|this\s+site)',
            r'by\s+continuing\s+to\s+browse\s+without',
            r'changing\s+your\s+browser\s+settings\s+to\s+block\s+or\s+delete\s+cookies',
            r'storing\s+of\s+cookies\s+and\s+related\s+technologies',
            r'first-party\s+cookies',
            r'third-party\s+cookies',
            r'strictly\s+necessary\s+cookies',
            r'performance\s+cookies',
            r'functional\s+cookies',
            r'targeting\s+cookies',
            r'advertising\s+cookies',
            r'marketing\s+cookies',
            r'cookie\s+consent',
            r'cookie\s+preferences',
            r'manage\s+cookies',
            r'cookie\s+settings',
            r'about\s+cookies',
            r'close\s+cookie\s+notice',
            r'link\s+opens\s+in\s+a\s+new\s+window',
            r'opens\s+in\s+new\s+window',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'terms\s+and\s+conditions',
            r'cookie\s+policy\s+link',
            r'gdpr',
            r'ccpa',
            r'data\s+protection',
            r'personal\s+data',
            r'data\s+collection',
            r'opt\s*[-]?\s*(?:in|out)',
            r'save\s+preferences',
            r'customize\s+(?:your\s+)?(?:cookie\s+)?preferences',
            r'always\s+active',
            r'allow\s+cookies',
            r'block\s+cookies',
            r'dismiss',
            r'got\s+it',
        ]
        
        # Common navigation/boilerplate patterns
        self.navigation_patterns = [
            r'^\s*(?:home|about|contact|menu|search|login|sign\s+in|sign\s+up)\s*$',
            r'breadcrumb',
            r'you\s+are\s+here',
            r'current\s+page',
            r'skip\s+to\s+(?:main\s+)?content',
            r'skip\s+navigation',
        ]
        
        # Common footer/header patterns
        self.footer_header_patterns = [
            r'copyright\s+©?\s*\d{4}',
            r'all\s+rights\s+reserved',
            r'follow\s+us\s+on',
            r'share\s+on',
            r'social\s+media',
            r'subscribe\s+to\s+our\s+newsletter',
            r'newsletter\s+signup',
        ]
        
        # Common error/empty content patterns
        self.error_patterns = [
            r'page\s+not\s+found',
            r'error\s+404',
            r'access\s+denied',
            r'forbidden',
            r'this\s+page\s+could\s+not\s+be\s+found',
            r'the\s+requested\s+page\s+does\s+not\s+exist',
        ]
        
        # Common UI/interface text
        self.ui_patterns = [
            r'click\s+here',
            r'read\s+more',
            r'show\s+more',
            r'show\s+less',
            r'load\s+more',
            r'view\s+all',
            r'see\s+more',
            r'download\s+pdf',
            r'print\s+this\s+page',
            r'share\s+this\s+page',
        ]
        
        # Common advertisement patterns
        self.ad_patterns = [
            r'advertisement',
            r'sponsored\s+content',
            r'promoted',
            r'ad\s+by',
        ]
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # LLM cleaner instance (lazy-loaded)
        self._llm_cleaner = None
    
    def _compile_patterns(self):
        """Compile all regex patterns for better performance"""
        # Use re.IGNORECASE flag instead of (?i) in patterns to avoid issues when joining
        self.cookie_regex = re.compile('|'.join(self.cookie_patterns), re.IGNORECASE)
        self.navigation_regex = re.compile('|'.join(self.navigation_patterns), re.IGNORECASE)
        self.footer_header_regex = re.compile('|'.join(self.footer_header_patterns), re.IGNORECASE)
        self.error_regex = re.compile('|'.join(self.error_patterns), re.IGNORECASE)
        self.ui_regex = re.compile('|'.join(self.ui_patterns), re.IGNORECASE)
        self.ad_regex = re.compile('|'.join(self.ad_patterns), re.IGNORECASE)
    
    def _has_cookie_content(self, text: str) -> bool:
        """
        Check if text contains cookie-related content that might need LLM cleaning
        
        Args:
            text: Text to check
        
        Returns:
            True if cookie content is detected
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for cookie-related keywords
        cookie_indicators = [
            'cookie', 'cookies', 'cookie policy', 'cookie notice', 'cookie consent',
            'strictly necessary cookies', 'performance cookies', 'functional cookies',
            'targeting cookies', 'about cookies', 'cookie information',
            'university of illinois system cookie policy',
            'always active', 'cookies and related technologies'
        ]
        
        # Count how many cookie indicators are present
        indicator_count = sum(1 for indicator in cookie_indicators if indicator in text_lower)
        
        # If multiple indicators or specific patterns, likely has cookie content
        if indicator_count >= 2:
            return True
        
        # Check for cookie policy structure patterns
        cookie_structure_patterns = [
            r'strictly\s+necessary\s+cookies\s*[-–]\s*always\s+active',
            r'performance\s+cookies\s*[-–]\s*always\s+active',
            r'functional\s+cookies\s*[-–]\s*always\s+active',
            r'targeting\s+cookies\s*[-–]\s*always\s+active',
        ]
        
        for pattern in cookie_structure_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

    def _has_llm_noise(self, text: str) -> bool:
        """
        Detect non-cookie noise that benefits from LLM cleaning
        (HTML/CSS markup, prompt artifacts, dataset markers, base64/garbled strings)
        """
        if not text:
            return False

        text_lower = text.lower()

        # Raw HTML/XML tags or obvious markup
        if re.search(r'<[^>]+>', text):
            return True

        # Inline style/attribute or CSS tokens
        if re.search(r'\b(style=|class=|id=|font-size|padding|margin|text-align)\b', text_lower):
            return True

        # Dataset markers / navigation artifacts
        marker_phrases = [
            "=== seed url",
            "=== profile page",
            "=== webpage",
            "top of page",
            "bottom of page",
            "back to top",
            "skip to content",
        ]
        if any(p in text_lower for p in marker_phrases):
            return True

        # Prompt/instruction artifacts
        prompt_phrases = [
            "critical rules",
            "output format",
            "text segment to analyze",
            "json output",
            "return only valid json",
        ]
        if any(p in text_lower for p in prompt_phrases):
            return True

        # Base64/hashed/garbled strings
        if re.search(r'[A-Za-z0-9+/]{50,}={0,2}', text):
            return True
        if re.search(r'\b[a-f0-9]{40,}\b', text_lower):
            return True

        return False
    
    def _get_llm_cleaner(self):
        """Lazy-load LLM cleaner if available"""
        if self._llm_cleaner is None:
            try:
                from api.utils.llm_text_cleaner import get_llm_text_cleaner
                self._llm_cleaner = get_llm_text_cleaner()
            except Exception as e:
                print(f"[TextCleaner] LLM cleaner not available: {str(e)}")
                self._llm_cleaner = False  # Mark as unavailable
        return self._llm_cleaner if self._llm_cleaner else None
    
    def clean_text(self, text: str, aggressive: bool = False, use_llm: bool = True) -> str:
        """
        Clean text content by removing noise, boilerplate, and irrelevant content
        
        Args:
            text: Raw text to clean
            aggressive: If True, apply more aggressive cleaning (may remove some valid content)
            use_llm: If True, use LLM cleaning as additional layer when cookie content is detected
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        raw_text = text
        
        # Step 0.5: Early cookie removal - remove obvious cookie blocks before other processing
        # This catches cookie content that might be embedded in HTML or special characters
        text = self._remove_cookie_blocks(text)
        
        # Step 1: Decode HTML entities
        text = html.unescape(text)
        
        # Step 1.5: Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Remove __bold__
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove *italic*
        text = re.sub(r'_([^_]+)_', r'\1', text)  # Remove _italic_
        text = re.sub(r'#{1,6}\s+', '', text)  # Remove markdown headers
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove markdown links, keep text
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove inline code
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # Remove code blocks
        
        # Step 2: Remove HTML tags and artifacts (in case any remain)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Step 2.5: Remove unwanted symbols and special characters (more aggressive)
        if aggressive:
            # Remove excessive punctuation and symbols
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"]+', ' ', text)  # Keep only common punctuation
            # Remove standalone symbols
            text = re.sub(r'\s+[^\w\s]{1,2}\s+', ' ', text)  # Remove single/double symbol words
        
        # Step 3: Convert line breaks to paragraph structure
        # First, replace multiple line breaks with paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        # Convert single line breaks within sentences to spaces (but keep paragraph breaks)
        # This is a simple approach - more complex logic in aggressive mode
        if aggressive:
            # In aggressive mode, convert all single \n to spaces, keep double \n as paragraph breaks
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single \n to space (not preceded/followed by \n)
        
        # Step 4: Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Ensure paragraph breaks are clean
        text = text.strip()
        
        # Step 5: Remove lines matching common noise patterns
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip very short lines that are likely navigation/UI elements
            if len(line) < 3:
                continue
            
            # Skip lines matching cookie/privacy patterns
            if self.cookie_regex.search(line):
                continue
            
            # Additional aggressive cookie detection: skip if line contains multiple cookie-related words
            cookie_word_count = sum(1 for pattern in self.cookie_patterns if re.search(pattern, line, re.IGNORECASE))
            if cookie_word_count >= 1:  # If line contains 1+ cookie-related patterns, skip it (more aggressive)
                continue
            
            # Also check for common cookie-related phrases that might not match regex patterns
            cookie_phrases_lower = ['gdpr', 'ccpa', 'data protection', 'tracking', 'analytics',
                                    'browser settings', 'opt out', 'opt-in', 'opt in', 'opt-out',
                                    'accept all', 'reject all', 'save preferences', 'dismiss', 'got it']
            if any(phrase in line.lower() for phrase in cookie_phrases_lower):
                # Check if it's likely cookie-related by looking for context
                cookie_context = ['cookie', 'consent', 'privacy', 'preferences', 'settings', 'banner']
                if any(ctx in line.lower() for ctx in cookie_context):
                    continue
            
            # Skip lines matching navigation patterns
            if self.navigation_regex.search(line):
                continue
            
            # Skip lines matching footer/header patterns
            if self.footer_header_regex.search(line):
                continue
            
            # Skip lines matching error patterns
            if self.error_regex.search(line):
                continue
            
            # Skip lines matching UI patterns (if aggressive)
            if aggressive and self.ui_regex.search(line):
                continue
            
            # Skip lines matching ad patterns
            if self.ad_regex.search(line):
                continue
            
            # Skip lines that are mostly special characters or numbers
            if aggressive and re.match(r'^[^\w\s]{3,}$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Step 6: Rejoin lines (empty lines become paragraph breaks)
        text = '\n'.join(cleaned_lines)
        # Clean up multiple consecutive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 7: Remove cookie/privacy notice blocks (multi-line) - do this early and multiple times
        # Apply cookie removal multiple times to catch nested/overlapping patterns
        for _ in range(3):  # Run 3 times to catch all cookie content
            text = self._remove_cookie_blocks(text)
        
        # Step 8: Remove common boilerplate phrases
        text = self._remove_boilerplate_phrases(text)
        
        # Step 8.5: Final cookie pass - remove any remaining cookie-related content
        text = self._remove_cookie_blocks(text)
        
        # Step 9: Additional aggressive cleaning if requested
        if aggressive:
            # Remove lines with only numbers, symbols, or very short content
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    # Preserve paragraph breaks
                    if cleaned_lines and cleaned_lines[-1]:
                        cleaned_lines.append('')
                    continue
                # Skip lines that are mostly symbols or numbers
                if line and not re.match(r'^[\d\s\W]{3,}$', line):
                    # Skip very short lines that are likely noise
                    if len(line) >= 5 or (len(line) >= 3 and re.search(r'[a-zA-Z]', line)):
                        cleaned_lines.append(line)
            text = '\n'.join(cleaned_lines)
            
            # Remove excessive line breaks again after filtering
            text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 10: Final paragraph structure normalization
        # Convert remaining single line breaks to spaces (except paragraph breaks)
        if aggressive:
            # Split by paragraph breaks, then join sentences within paragraphs
            paragraphs = text.split('\n\n')
            cleaned_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if para:
                    # Within a paragraph, replace single \n with space
                    para = re.sub(r'\n+', ' ', para)
                    # Normalize spaces
                    para = re.sub(r'\s+', ' ', para)
                    cleaned_paragraphs.append(para)
            text = '\n\n'.join(cleaned_paragraphs)
        
        # Final whitespace normalization
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple paragraph breaks to double
        text = text.strip()
        
        # Step 11: Optional LLM cleaning layer for cookie or noisy content
        # Use LLM as additional layer if cookie content or noisy artifacts are detected
        if use_llm and text:
            llm_cleaner = self._get_llm_cleaner()
            if llm_cleaner and (self._has_cookie_content(raw_text) or self._has_llm_noise(raw_text)):
                try:
                    print(f"[TextCleaner] Detected cookie/noise content, applying LLM cleaning...")
                    # Use LLM to clean the text (it handles chunking internally)
                    llm_cleaned = llm_cleaner.clean_text(text, timeout=30, use_chunking=True)
                    
                    # Only use LLM result if it's reasonable (not too short, not empty)
                    if llm_cleaned and len(llm_cleaned) > len(text) * 0.1:
                        print(f"[TextCleaner] ✅ LLM cleaning applied: {len(text)} -> {len(llm_cleaned)} chars")
                        text = llm_cleaned
                    else:
                        print(f"[TextCleaner] LLM result too short, keeping regex-cleaned text")
                except Exception as e:
                    print(f"[TextCleaner] LLM cleaning failed: {str(e)}, using regex-cleaned text")
                    # Continue with regex-cleaned text if LLM fails
        
        return text
    
    def _remove_cookie_blocks(self, text: str) -> str:
        """Remove multi-line cookie/privacy notice blocks - more aggressive approach"""
        if not text:
            return text
        
        # First, split text into lines for more precise removal
        lines = text.split('\n')
        cleaned_lines = []
        
        # Expanded cookie keywords - including variations and related terms
        cookie_keywords = [
            'cookie', 'cookies', 'cookie policy', 'cookie notice', 'cookie consent',
            'cookie information', 'cookie settings', 'cookie preferences',
            'privacy policy', 'terms of service', 'terms and conditions',
            'accept cookies', 'agree to cookies', 'cookie banner',
            'strictly necessary cookies', 'performance cookies', 'functional cookies',
            'targeting cookies', 'first-party cookies', 'third-party cookies',
            'we use cookies', 'this website uses cookies', 'by continuing',
            'link opens in a new window', 'about cookies', 'close cookie',
            'cookie consent', 'manage cookies', 'cookie preferences',
            'gdpr', 'ccpa', 'data protection', 'tracking', 'analytics',
            'personal data', 'data collection', 'browser settings',
            'opt out', 'opt-in', 'opt in', 'opt-out',
            # University of Illinois specific patterns
            'university of illinois system cookie policy',
            'strictly necessary cookies - always active',
            'performance cookies - always active',
            'functional cookies - always active',
            'targeting cookies - always active',
            'cookies and related technologies',
            'cookies the university sets',
            'cookies set by third parties'
        ]
        
        # Context words that often appear in cookie notices (even without "cookie" keyword)
        cookie_context_words = [
            'browser settings', 'tracking', 'analytics', 'collecting data',
            'aggregating', 'anonymous', 'third-party', 'first-party',
            'personal information', 'data protection', 'privacy settings',
            'preferences', 'consent', 'accept all', 'reject all',
            'necessary', 'functional', 'performance', 'advertising',
            'marketing', 'social media', 'embedded content',
            'session', 'persistent', 'local storage', 'web storage',
            # Additional patterns for cookie policy sections
            'always active', 'click on', 'about cookies', 'close cookie notice',
            'university of illinois', 'cookie information', 'manually adjust preferences',
            'block or delete cookies', 'change your browser settings'
        ]
        
        # Common cookie-related phrases
        cookie_phrases = [
            'accept all', 'reject all', 'save preferences', 'customize',
            'always active', 'allow cookies', 'block cookies',
            'cookie banner', 'cookie popup', 'cookie dialog'
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_lower = line.lower()
            
            # Check if this line contains cookie-related keywords
            has_cookie_keyword = any(keyword in line_lower for keyword in cookie_keywords)
            
            # Also check for context words that suggest cookie content
            has_cookie_context = any(context in line_lower for context in cookie_context_words)
            
            # Check if line contains common cookie-related phrases
            has_cookie_phrase = any(phrase in line_lower for phrase in cookie_phrases)
            
            if has_cookie_keyword or (has_cookie_context and has_cookie_phrase):
                # Skip this line and potentially following lines
                # Look ahead to find the end of the cookie block - more aggressive approach
                block_end = i
                cookie_line_count = 0
                consecutive_non_cookie = 0
                max_consecutive_non_cookie = 3  # Allow up to 3 consecutive non-cookie lines before stopping
                
                # Look ahead up to 100 lines for cookie-related content (increased from 50)
                for j in range(i, min(i + 100, len(lines))):
                    check_line = lines[j].strip().lower()
                    
                    # Check if this line is cookie-related
                    check_has_keyword = any(keyword in check_line for keyword in cookie_keywords)
                    check_has_context = any(context in check_line for context in cookie_context_words)
                    check_has_phrase = any(phrase in check_line for phrase in cookie_phrases)
                    
                    # Also check for common cookie notice endings
                    cookie_endings = ['close', 'accept', 'agree', 'continue', 'save', 'ok', 'got it', 'dismiss']
                    has_ending = any(ending in check_line for ending in cookie_endings)
                    
                    if check_has_keyword or (check_has_context and check_has_phrase) or has_ending:
                        cookie_line_count += 1
                        block_end = j
                        consecutive_non_cookie = 0  # Reset counter
                    else:
                        consecutive_non_cookie += 1
                        # If we've seen enough cookie lines and hit several non-cookie lines, stop
                        if cookie_line_count >= 2 and consecutive_non_cookie >= max_consecutive_non_cookie:
                            # But check if the next few lines might still be cookie-related
                            look_ahead_lines = min(3, len(lines) - j - 1)
                            found_more_cookie = False
                            for k in range(1, look_ahead_lines + 1):
                                if j + k < len(lines):
                                    ahead_line = lines[j + k].strip().lower()
                                    if any(keyword in ahead_line for keyword in cookie_keywords):
                                        found_more_cookie = True
                                        break
                            if not found_more_cookie:
                                break
                        # Still extend block_end if we're in a potential cookie section
                        elif cookie_line_count > 0:
                            block_end = j
                
                # Skip the entire cookie block
                i = block_end + 1
                continue
            
            cleaned_lines.append(lines[i])
            i += 1
        
        text = '\n'.join(cleaned_lines)
        
        # Now apply regex patterns for remaining cookie content - expanded and more comprehensive
        patterns = [
            # Cookie policy/notice blocks (more flexible patterns)
            r'click\s+on\s+["\']?about\s+cookies["\']?.{0,5000}?close',
            r'cookie\s+notice.{0,5000}?(?:close|accept|agree|continue|dismiss|got\s+it)',
            r'cookie\s+information.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'about\s+cookies.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'university\s+of\s+.*?\s+cookie\s+policy.{0,5000}?(?:close|accept|agree|continue)',
            r'(?:cookie|privacy|terms|consent|accept|gdpr|ccpa).{0,200}?\n.{0,5000}?(?:close|accept|agree|continue|settings|preferences|dismiss|got\s+it)',
            r'we\s+use\s+cookies?.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'by\s+continuing.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'this\s+website\s+uses\s+cookies?.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'click\s+(?:here|accept|ok)\s+to\s+(?:accept|agree|continue).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'(?:accept|agree)\s+(?:all\s+)?cookies?.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'cookie\s+(?:policy|notice|settings|preferences|information|banner|popup|dialog).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            # Cookie categories with descriptions (more comprehensive)
            r'strictly\s+necessary\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'performance\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'functional\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'targeting\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'advertising\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'marketing\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            # Cookie technology descriptions
            r'cookies\s+and\s+related\s+technologies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'first-party\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'third-party\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'cookies\s+the\s+(?:university|site|website|organization)\s+sets.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'cookies\s+set\s+by\s+third\s+parties.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            # Browser settings
            r'change\s+your\s+browser\s+settings.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'block\s+or\s+delete\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'manually\s+adjust\s+preferences.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'browser\s+settings\s+to\s+(?:block|delete|manage)\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            # Analytics and tracking
            r'google\s+analytics.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'analytics\s+cookies.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'collecting\s+and\s+aggregating\s+anonymous\s+information.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            r'tracking\s+(?:technologies|tools|cookies).{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|close|accept|agree|$)',
            # GDPR/Privacy related
            r'gdpr.{0,200}?(?:cookie|consent|privacy).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'data\s+protection.{0,200}?(?:cookie|consent).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'personal\s+data.{0,200}?(?:cookie|collection).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            # Opt-in/opt-out
            r'opt\s*[-]?\s*(?:in|out).{0,200}?(?:cookie|tracking).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            # Link text artifacts
            r'link\s+opens\s+in\s+a\s+new\s+window.{0,200}?(?:\n|$)',
            r'opens\s+in\s+new\s+window.{0,200}?(?:\n|$)',
            # Common cookie banner phrases
            r'(?:accept\s+all|reject\s+all|save\s+preferences|customize).{0,5000}?(?:close|accept|agree|continue|dismiss)',
            r'always\s+active.{0,5000}?(?:close|accept|agree|continue|dismiss)',
            # University of Illinois specific patterns - catch entire cookie policy sections
            r'click\s+on\s+["\']?about\s+cookies["\']?.{0,10000}?close',
            r'university\s+of\s+illinois\s+system\s+cookie\s+policy.{0,10000}?close',
            r'university\s+of\s+illinois\s*[-–]\s*cookie\s+information.{0,10000}?close',
            r'strictly\s+necessary\s+cookies\s*[-–]\s*always\s+active.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|$)',
            r'performance\s+cookies\s*[-–]\s*always\s+active.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|$)',
            r'functional\s+cookies\s*[-–]\s*always\s+active.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|$)',
            r'targeting\s+cookies\s*[-–]\s*always\s+active.{0,5000}?(?:\n\n|\n[A-Z][a-z]{3,}|$)',
            # Catch cookie policy sections that start with "About Cookies" or similar
            r'about\s+cookies.{0,200}?cookies\s+and\s+related\s+technologies.{0,10000}?(?:strictly\s+necessary|performance|functional|targeting|close|$)',
            # Catch entire cookie policy blocks
            r'(?:click\s+on|by\s+continuing|university\s+of\s+illinois).{0,200}?(?:cookie|browse).{0,500}?(?:strictly\s+necessary|performance|functional|targeting|always\s+active).{0,15000}?(?:close|$)',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Remove cookie policy sections that span multiple paragraphs (more aggressive)
        cookie_section_patterns = [
            r'(?:cookie|privacy|consent|accept\s+cookies|gdpr|data\s+protection).{0,100}?(?:\n.{0,200}?){3,150}?(?:close|accept|agree|continue|settings|preferences|always\s+active|dismiss|got\s+it)',
            r'cookie\s+(?:banner|popup|dialog|notice).{0,5000}?(?:close|accept|agree|continue|dismiss|got\s+it)',
            # Catch University of Illinois cookie policy format - entire section from start to end
            r'click\s+on\s+["\']?about\s+cookies["\']?.{0,500}?(?:strictly\s+necessary|performance|functional|targeting|always\s+active).{0,20000}?(?:close|$)',
            r'university\s+of\s+illinois.{0,300}?(?:cookie|policy|information).{0,500}?(?:strictly\s+necessary|performance|functional|targeting|always\s+active).{0,20000}?(?:close|$)',
            # Catch cookie category sections with "Always Active" pattern
            r'(?:strictly\s+necessary|performance|functional|targeting)\s+cookies\s*[-–]\s*always\s+active.{0,3000}?(?:(?:strictly\s+necessary|performance|functional|targeting)\s+cookies\s*[-–]\s*always\s+active|close|$)',
        ]
        for pattern in cookie_section_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        # Remove any remaining lines that are mostly cookie-related (more aggressive filtering)
        lines = text.split('\n')
        final_lines = []
        all_cookie_terms = cookie_keywords + cookie_context_words + cookie_phrases
        
        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                final_lines.append('')
                continue
            
            # Count cookie-related terms in the line
            cookie_term_count = sum(1 for term in all_cookie_terms if term in line_lower)
            total_words = len(line_lower.split())
            
            # Skip lines that are primarily cookie-related (lowered threshold from 30% to 20%)
            if total_words > 0 and cookie_term_count > 0:
                cookie_ratio = cookie_term_count / max(total_words, 1)
                if cookie_ratio > 0.2:  # More than 20% cookie terms
                    continue
            
            # Skip lines that are very short and contain cookie keywords
            if total_words < 8 and cookie_term_count > 0:  # Increased from 5 to 8 words
                continue
            
            # Skip lines that are just cookie-related phrases
            if cookie_term_count >= 2 and total_words < 15:
                continue
            
            final_lines.append(line)
        
        text = '\n'.join(final_lines)
        
        # Final cleanup: remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _remove_boilerplate_phrases(self, text: str) -> str:
        """Remove common boilerplate phrases from text"""
        boilerplate_phrases = [
            r'\b(?:last\s+updated|updated\s+on|published\s+on)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:page\s+\d+\s+of\s+\d+)\b',
            r'\b(?:showing\s+\d+\s+to\s+\d+\s+of\s+\d+)\b',
            r'\b(?:results?\s+\d+\s+to\s+\d+)\b',
        ]
        
        for phrase in boilerplate_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        return text
    
    def clean_structured_content(self, content: Dict) -> Dict:
        """
        Clean structured content (full_text, headings, paragraphs)
        
        Args:
            content: Dictionary with 'full_text', 'headings', 'paragraphs' keys
        
        Returns:
            Cleaned content dictionary
        """
        cleaned = {}
        
        if 'full_text' in content:
            cleaned['full_text'] = self.clean_text(content['full_text'])
        
        if 'headings' in content and isinstance(content['headings'], list):
            cleaned_headings = []
            for heading in content['headings']:
                if isinstance(heading, str):
                    cleaned_heading = self.clean_text(heading, aggressive=True)
                    if cleaned_heading and len(cleaned_heading) > 2:
                        cleaned_headings.append(cleaned_heading)
            cleaned['headings'] = cleaned_headings
        
        if 'paragraphs' in content and isinstance(content['paragraphs'], list):
            cleaned_paragraphs = []
            for para in content['paragraphs']:
                if isinstance(para, str):
                    cleaned_para = self.clean_text(para)
                    if cleaned_para and len(cleaned_para) > 10:  # Minimum paragraph length
                        cleaned_paragraphs.append(cleaned_para)
            cleaned['paragraphs'] = cleaned_paragraphs
        
        return cleaned
    
    def clean_document_content(self, content: str) -> str:
        """
        Clean content extracted from documents (PDFs, Word docs, etc.)
        
        Args:
            content: Raw document text
        
        Returns:
            Cleaned document text
        """
        if not content:
            return ""
        
        # Apply standard cleaning WITHOUT LLM (documents are usually clean, LLM is slow)
        # Use aggressive=False and use_llm=False for faster processing
        cleaned = self.clean_text(content, aggressive=False, use_llm=False)
        
        # Additional document-specific cleaning
        # Remove page numbers (usually at start/end of lines)
        cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)
        
        # Remove common PDF artifacts
        cleaned = re.sub(r'\f', '\n', cleaned)  # Form feed to newline
        cleaned = re.sub(r'\x0c', '\n', cleaned)  # Form feed character
        
        # Remove excessive line breaks (more than 2 consecutive)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()


# Singleton instance
_text_cleaner_instance = None

def get_text_cleaner() -> TextCleaner:
    """Get or create TextCleaner instance"""
    global _text_cleaner_instance
    if _text_cleaner_instance is None:
        _text_cleaner_instance = TextCleaner()
    return _text_cleaner_instance


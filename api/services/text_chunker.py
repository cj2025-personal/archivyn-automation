"""
Text chunking service using LangChain's RecursiveCharacterTextSplitter
Chunks text while preserving section/heading information
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional
import re


class TextChunker:
    """Chunk text using RecursiveCharacterTextSplitter with section awareness"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = None):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters). If None, uses 10% of chunk_size
        """
        self.chunk_size = chunk_size
        # Default to 10% overlap if not specified
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(chunk_size * 0.1)
        # Note: We'll create a new splitter after cleaning text (without newlines)
        # So separators should focus on sentence and word boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[". ", "! ", "? ", "; ", ", ", " ", ""]  # Sentence, clause, word boundaries for semantic chunking
        )
    
    def chunk_text_with_sections(self, text: str, headings: List[str] = None, 
                                 source_sections: List[Dict] = None, 
                                 use_llm_for_sections: bool = True,
                                 professor_name: str = None,
                                 llm_clean_chunks: bool = True,
                                 llm_timeout: int = 12) -> List[Dict]:
        """
        Remove newlines, create semantically coherent chunks section-wise with 10% overlap
        Automatically detects sections from text using LLM
        
        Args:
            text: Text to chunk
            headings: List of headings found in the text (optional, will be auto-detected if None)
            source_sections: List of section metadata (e.g., from different sources)
            use_llm_for_sections: Whether to use LLM for section detection (default: True)
            professor_name: Name of the professor (to be assigned to each chunk)
        
        Returns:
            List of chunk dictionaries with section information, chunked section-wise
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Detect sections from text using LLM (if not provided)
        detected_sections = []
        if not source_sections or len(source_sections) == 0:
            try:
                from api.services.section_detector import get_section_detector
                section_detector = get_section_detector(use_llm=use_llm_for_sections)
                detected_sections = section_detector.detect_sections(text, use_llm=use_llm_for_sections)
                print(f"[TextChunker] Detected {len(detected_sections)} sections using {'LLM' if use_llm_for_sections else 'pattern'}: {[s.get('title', '') for s in detected_sections]}")
            except Exception as e:
                print(f"[TextChunker] Section detection failed: {str(e)}, using provided sections")
                import traceback
                traceback.print_exc()
        
        # Use detected sections or provided source_sections
        sections_to_use = detected_sections if detected_sections else source_sections
        
        # Step 2: Chunk each section separately with 10% overlap
        all_chunks = []
        chunk_index = 0
        
        if sections_to_use and len(sections_to_use) > 0:
            # Sort sections by start position
            sections_to_use = sorted(sections_to_use, key=lambda x: x.get('start', 0))
            
            for section_idx, section in enumerate(sections_to_use):
                section_start = section.get('start', 0)
                section_end = section.get('end', len(text))
                section_title = section.get('title', section.get('type', f'Section {section_idx + 1}'))
                section_type = section.get('type', 'other')
                
                # Extract section text
                section_text = text[section_start:section_end]
                
                if not section_text.strip():
                    continue
                
                # Clean section text
                cleaned_section_text = self._clean_chunk_text(section_text)
                
                # Chunk this section with 10% overlap
                section_chunks = self._create_semantic_chunks(cleaned_section_text)
                
                # Annotate chunks with section information
                for chunk in section_chunks:
                    if llm_clean_chunks:
                        chunk = self._clean_chunk_with_llm(chunk, professor_name, section_title, timeout=llm_timeout)
                        if not chunk.strip():
                            continue
                    # Map chunk position back to original text
                    chunk_pos_in_section = cleaned_section_text.find(chunk)
                    original_chunk_start = section_start + chunk_pos_in_section if chunk_pos_in_section >= 0 else section_start
                    
                    annotated_chunk = {
                        'chunk_id': f"chunk_{chunk_index}",
                        'chunk_index': chunk_index,
                        'text': chunk,  # Already cleaned (no newlines)
                        'char_count': len(chunk),
                        'word_count': len(chunk.split()),
                        'professor_name': professor_name or '',
                        'section': section_title,
                        'heading': section_title,
                        'section_type': section_type,
                        'source': section.get('method', section.get('source', 'detected')),
                        'start_position': original_chunk_start,
                        'end_position': original_chunk_start + len(chunk) if original_chunk_start >= 0 else section_end
                    }
                    
                    all_chunks.append(annotated_chunk)
                    chunk_index += 1
        else:
            # No sections detected, chunk entire text as one section
            cleaned_text = self._clean_chunk_text(text)
            chunks = self._create_semantic_chunks(cleaned_text)
            
            for idx, chunk in enumerate(chunks):
                if llm_clean_chunks:
                    chunk = self._clean_chunk_with_llm(chunk, professor_name, '', timeout=llm_timeout)
                    if not chunk.strip():
                        continue
                chunk_start_in_cleaned = cleaned_text.find(chunk)
                original_pos = self._map_cleaned_to_original_position(text, cleaned_text, chunk_start_in_cleaned)
                
                annotated_chunk = {
                    'chunk_id': f"chunk_{idx}",
                    'chunk_index': idx,
                    'text': chunk,
                    'char_count': len(chunk),
                    'word_count': len(chunk.split()),
                    'professor_name': professor_name or '',
                    'section': '',
                    'heading': '',
                    'section_type': 'other',
                    'source': 'no_sections',
                    'start_position': original_pos,
                    'end_position': original_pos + len(chunk) if original_pos >= 0 else 0
                }
                
                all_chunks.append(annotated_chunk)
        
        return all_chunks
    
    def _build_heading_map(self, text: str, headings: List[str] = None) -> Dict[int, str]:
        """
        Build a map of text positions to headings
        
        Returns:
            Dictionary mapping position to heading text
        """
        heading_map = {}
        
        if not headings:
            # Try to extract headings from text using common patterns
            heading_patterns = [
                r'^=== (.+?) ===',  # === SECTION ===
                r'^#+\s+(.+?)$',     # Markdown headings
                r'^([A-Z][A-Z\s]{10,})$',  # ALL CAPS headings
            ]
            
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                for pattern in heading_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        heading_text = match.group(1).strip()
                        position = sum(len(l) + 1 for l in lines[:line_num])
                        heading_map[position] = heading_text
                        break
        else:
            # Use provided headings and find their positions in text
            for heading in headings:
                if heading:
                    position = text.find(heading)
                    if position >= 0:
                        heading_map[position] = heading
        
        return heading_map
    
    def _build_section_map(self, text: str, source_sections: List[Dict] = None) -> Dict[int, Dict]:
        """
        Build a map of text positions to source sections
        
        Args:
            text: Full text
            source_sections: List of section dictionaries with 'start', 'end', 'type', 'source' keys
        
        Returns:
            Dictionary mapping position to section info
        """
        section_map = {}
        
        if source_sections:
            for section in source_sections:
                start = section.get('start', 0)
                section_type = section.get('type', '')
                source = section.get('source', '')
                section_map[start] = {
                    'type': section_type,
                    'source': source,
                    'end': section.get('end', len(text))
                }
        
        return section_map
    
    def _find_section_for_chunk(self, chunk_position: int, heading_map: Dict[int, str],
                                section_map: Dict[int, Dict], full_text: str) -> Dict:
        """
        Find the section/heading that a chunk belongs to
        
        Returns:
            Dictionary with section information
        """
        result = {
            'section': '',
            'heading': '',
            'section_type': '',
            'source': ''
        }
        
        # Find the closest heading before this chunk
        closest_heading_pos = -1
        closest_heading = ''
        
        for pos, heading in heading_map.items():
            if pos <= chunk_position and pos > closest_heading_pos:
                closest_heading_pos = pos
                closest_heading = heading
        
        if closest_heading:
            result['heading'] = closest_heading
        
        # Find the section this chunk belongs to
        closest_section_pos = -1
        closest_section = {}
        
        for pos, section_info in section_map.items():
            section_end = section_info.get('end', len(full_text))
            if pos <= chunk_position < section_end and pos > closest_section_pos:
                closest_section_pos = pos
                closest_section = section_info
        
        if closest_section:
            result['section_type'] = closest_section.get('type', '')
            result['source'] = closest_section.get('source', '')
            # Use section type as section name if no heading found
            if not result['heading']:
                result['section'] = result['section_type']
        
        # If we found a heading but no section type, use heading as section
        if result['heading'] and not result['section']:
            result['section'] = result['heading']
        
        # Try to detect section from text patterns around chunk position
        if not result['section']:
            # Look for section markers before chunk
            context_start = max(0, chunk_position - 200)
            context = full_text[context_start:chunk_position]
            
            # Check for section markers like "=== SECTION ==="
            section_match = re.search(r'=== ([^=]+) ===', context)
            if section_match:
                result['section'] = section_match.group(1).strip()
                result['section_type'] = 'document_section'
        
        return result
    
    def _map_cleaned_to_original_position(self, original_text: str, cleaned_text: str, cleaned_pos: int) -> int:
        """
        Map a position in cleaned text back to original text position
        Accounts for removed newlines
        
        Args:
            original_text: Original text with newlines
            cleaned_text: Cleaned text without newlines
            cleaned_pos: Position in cleaned text
        
        Returns:
            Approximate position in original text
        """
        if cleaned_pos >= len(cleaned_text):
            return len(original_text)
        
        # Count characters up to cleaned_pos in cleaned text
        # Then find corresponding position in original text
        # by matching character sequences
        
        # Simple approach: find the substring in original text
        # that corresponds to the cleaned substring
        if cleaned_pos == 0:
            return 0
        
        # Get the text up to this position in cleaned text
        cleaned_prefix = cleaned_text[:cleaned_pos]
        
        # Find this prefix in original text (accounting for newlines)
        # We'll search for a pattern that matches when newlines are removed
        import re
        # Create a regex pattern that matches the cleaned prefix
        # allowing for newlines to be anywhere
        pattern = re.escape(cleaned_prefix[-50:]) if len(cleaned_prefix) > 50 else re.escape(cleaned_prefix)
        pattern = pattern.replace(r'\ ', r'[\s\n\r]+')
        
        # Try to find the position
        match = re.search(pattern, original_text)
        if match:
            return match.end() - len(cleaned_prefix[-50:]) if len(cleaned_prefix) > 50 else match.end() - len(cleaned_prefix)
        
        # Fallback: estimate based on character ratio
        if len(cleaned_text) > 0:
            ratio = len(original_text) / len(cleaned_text)
            return int(cleaned_pos * ratio)
        
        return 0
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """
        Create semantically coherent chunks by:
        1. Splitting into sentences
        2. Grouping sentences into chunks that don't break mid-thought
        3. Ensuring chunks are within size limits
        
        Args:
            text: Cleaned text (no newlines)
        
        Returns:
            List of semantically coherent chunk texts
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences using multiple delimiters
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap (last few sentences from previous chunk)
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s) + 1  # +1 for space
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences intelligently, handling abbreviations
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        import re
        
        # Common abbreviations that shouldn't end sentences
        # Include academic degrees, titles, common abbreviations
        abbr_patterns = [
            r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Ph\.D|M\.S|B\.S|B\.A|M\.A|M\.B\.A|Ph\.D|etc|i\.e|e\.g|vs|vs\.|Inc|Ltd|Corp|Co|St|Ave|Rd|Blvd|U\.S|U\.K|U\.N)\.',
            r'\b([A-Z]\.)',  # Single capital letter followed by period (like "U.S.")
            r'\b\d{4}\.',  # Years followed by period (like "2013.")
        ]
        
        # Replace abbreviations with placeholders to avoid splitting on them
        text_processed = text
        abbr_map = {}
        abbr_counter = 0
        
        for pattern in abbr_patterns:
            matches = list(re.finditer(pattern, text_processed, re.IGNORECASE))
            # Process in reverse to maintain positions
            for match in reversed(matches):
                placeholder = f"__ABBR_{abbr_counter}__"
                abbr_map[placeholder] = match.group(0)
                text_processed = text_processed[:match.start()] + placeholder + text_processed[match.end():]
                abbr_counter += 1
        
        # Split on sentence endings: . ! ? followed by space and capital letter or end of string
        # Use a simpler approach: find all sentence boundaries
        # Pattern: sentence ending (. ! ?) followed by space and capital letter OR end of string
        sentence_boundaries = []
        
        # Find all potential sentence endings
        for match in re.finditer(r'[.!?]\s+(?=[A-Z])|[.!?](?=\s*$)', text_processed):
            sentence_boundaries.append(match.end())
        
        # Split text at boundaries
        sentences = []
        start = 0
        
        for boundary in sentence_boundaries:
            sentence = text_processed[start:boundary].strip()
            if sentence:
                # Restore abbreviations
                for placeholder, original in abbr_map.items():
                    sentence = sentence.replace(placeholder, original)
                sentences.append(sentence)
            start = boundary
        
        # Add remaining text
        if start < len(text_processed):
            sentence = text_processed[start:].strip()
            if sentence:
                # Restore abbreviations
                for placeholder, original in abbr_map.items():
                    sentence = sentence.replace(placeholder, original)
                sentences.append(sentence)
        
        # Fallback: if no sentences found or too few, use simpler splitting
        if len(sentences) < 2:
            # Split by periods followed by space and capital letter
            sentences = re.split(r'\.\s+(?=[A-Z])', text)
            sentences = [s.strip() + '.' if s.strip() and not s.strip().endswith(('.', '!', '?')) else s.strip() 
                         for s in sentences if s.strip() and len(s.strip()) > 1]
        
        # Clean up sentences: remove empty ones and ensure they end properly
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 1:
                # Ensure sentence ends with punctuation
                if not sent[-1] in ('.', '!', '?', ':', ';'):
                    # Check if it's a complete thought (ends with word)
                    if sent[-1].isalnum():
                        sent += '.'
                cleaned_sentences.append(sent)
        
        return cleaned_sentences if cleaned_sentences else [text]
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean chunk text by removing newlines, unicode symbols, and normalizing whitespace
        Removes symbols exclusively while preserving text content
        
        Args:
            text: Raw chunk text
        
        Returns:
            Cleaned text with newlines, unicode symbols removed and whitespace normalized
        """
        if not text:
            return ''
        
        import re
        
        # Remove all newline characters (\n, \r\n, \r, \t)
        cleaned = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Remove common unicode bullet points and symbols
        # Bullet points: •, ·, ▪, ▫, ○, ●, ◘, ◙, ‣, ⁃, etc.
        unicode_bullets = [
            '\u2022',  # •
            '\u00b7',  # ·
            '\u25aa',  # ▪
            '\u25ab',  # ▫
            '\u25cb',  # ○
            '\u25cf',  # ●
            '\u25d8',  # ◘
            '\u25d9',  # ◙
            '\u2023',  # ‣
            '\u2043',  # ⁃
            '\u2219',  # ∙
            '\u22c5',  # ⋅
            '\u30fb',  # ・
            '\u25e6',  # ◦
            '\u25aa',  # ▪
            '\u25a0',  # ■
            '\u25a1',  # □
        ]
        
        for bullet in unicode_bullets:
            cleaned = cleaned.replace(bullet, ' ')
        
        # Remove other common unicode symbols that might interfere
        # Non-breaking spaces, zero-width spaces, etc.
        unicode_spaces = [
            '\u00a0',  # Non-breaking space
            '\u2000',  # En quad
            '\u2001',  # Em quad
            '\u2002',  # En space
            '\u2003',  # Em space
            '\u2004',  # Three-per-em space
            '\u2005',  # Four-per-em space
            '\u2006',  # Six-per-em space
            '\u2007',  # Figure space
            '\u2008',  # Punctuation space
            '\u2009',  # Thin space
            '\u200a',  # Hair space
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\u2028',  # Line separator
            '\u2029',  # Paragraph separator
            '\u202f',  # Narrow no-break space
            '\u205f',  # Medium mathematical space
            '\u3000',  # Ideographic space
            '\ufeff',  # Zero-width no-break space (BOM)
        ]
        
        for space in unicode_spaces:
            cleaned = cleaned.replace(space, ' ')
        
        # Remove various unicode symbol ranges (but keep letters, numbers, and basic punctuation)
        # Mathematical symbols, arrows, geometric shapes, box drawing, etc.
        cleaned = re.sub(r'[\u2000-\u206f\u2190-\u21ff\u2200-\u22ff\u2300-\u23ff\u2400-\u243f\u2440-\u245f\u2460-\u24ff\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf\u27c0-\u27ef\u27f0-\u27ff\u2800-\u28ff\u2900-\u297f\u2980-\u29ff\u2a00-\u2aff\u2b00-\u2bff\u2e00-\u2e7f\u3000-\u303f\ufeff]', ' ', cleaned)
        
        # Remove control characters (except space)
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        # Final cleanup: keep core punctuation and common profile symbols (emails, degrees, C++, C#, etc.)
        # while removing residual unsupported glyph noise.
        cleaned = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"\(\)\[\]\{\}\/@&%+#=:_]', ' ', cleaned)
        
        # Normalize multiple spaces to single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Strip leading/trailing whitespace
        return cleaned.strip()

    def _clean_chunk_with_llm(self, chunk: str, professor_name: str, section_title: str, timeout: int = 12) -> str:
        """
        Optionally clean a chunk with LLM to strip navigation, symbols, and boilerplate.
        Falls back to regex-based cleaning if LLM unavailable or fails.
        """
        if not chunk or not chunk.strip():
            return ''
        try:
            from api.utils.llm_text_cleaner import get_llm_text_cleaner

            cleaner = get_llm_text_cleaner()
            if not cleaner:
                return self._clean_chunk_text(chunk)

            system_prompt = (
                "You clean and rewrite small text excerpts from academic/staff profiles. "
                "REMOVE only obvious noise and Unicode artifacts. Preserve every factual detail.\n"
                "- Delete navigation/global menus, 'show submenu' strings, category/link lists (academics, admissions, alumni, careers, etc.), headers/footers, social/link icon lists, cookie/privacy/terms notices, and boilerplate UI text.\n"
                "- Delete raw HTML/XML/CSS/JS markup or attribute noise (tags, style/class attributes, font-size, padding, margin, text-align); keep only visible meaningful text if any.\n"
                "- Delete legal/policy sections (privacy/terms/consent/CCPA/California-resident notices) and cookie category lists.\n"
                "- Delete prompt/instruction artifacts and dataset markers (CRITICAL RULES, Output format, Text segment to analyze, JSON output, === SEED URL ===, === PROFILE PAGE ===, === WEBPAGE ===, top of page, bottom of page, back to top).\n"
                "- Delete base64/hashed/garbled strings and tracking IDs.\n"
                "- Delete long run-on nav menus and stray symbols/emojis/bullet characters; normalize whitespace.\n"
                "- Rewrite remaining text into clear, coherent prose while retaining ALL facts, names, titles, dates, institutions, and publication details.\n"
                "- Do NOT summarize, abstract, shorten, or omit details.\n"
                "- Do NOT add any new facts that are not explicitly present.\n"
                "- If the text is pure navigation/boilerplate after removal, return an empty string."
            )
            user_prompt = (
                f"Name/context: {professor_name or 'unknown'} | Section: {section_title or 'unspecified'}\n"
                f"Text:\n{chunk}"
            )

            response = cleaner.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=timeout,
            )

            cleaned = (response.choices[0].message.content or "").strip()
            cleaned = self._clean_chunk_text(cleaned)
            print(f"[TextChunker] Chunk cleaned via LLM (GPT) | section={section_title or 'n/a'}")
            return cleaned
        except Exception as e:
            print(f"[TextChunker] LLM chunk cleaning failed: {str(e)}, using regex cleanup")
            return self._clean_chunk_text(chunk)

    def chunk_structured_text(self, text: str, headings: List[str] = None,
                             paragraphs: List[str] = None, use_llm_for_sections: bool = False,
                             professor_name: str = None, llm_clean_chunks: bool = True,
                             llm_timeout: int = 12) -> List[Dict]:
        """
        Chunk structured text (with headings and paragraphs) preserving structure
        Removes newlines, creates semantic chunks, and maps to sections
        Automatically detects sections if not provided
        
        Args:
            text: Full text to chunk
            headings: List of headings
            paragraphs: List of paragraphs
            use_llm_for_sections: Whether to use LLM for section detection
            professor_name: Name of the professor (to be assigned to each chunk)
            llm_clean_chunks: Whether to LLM-clean each chunk before returning
            llm_timeout: Timeout for LLM cleaning calls in seconds
        
        Returns:
            List of annotated chunks (with newlines removed)
        """
        # Build source sections from headings and paragraphs
        source_sections = []
        
        if headings:
            current_pos = 0
            for heading in headings:
                heading_pos = text.find(heading, current_pos)
                if heading_pos >= 0:
                    # Find where this section ends (next heading or end of text)
                    next_heading_pos = len(text)
                    for next_heading in headings:
                        if next_heading != heading:
                            next_pos = text.find(next_heading, heading_pos + len(heading))
                            if next_pos >= 0 and next_pos < next_heading_pos:
                                next_heading_pos = next_pos
                    
                    source_sections.append({
                        'start': heading_pos,
                        'end': next_heading_pos,
                        'type': heading,
                        'source': 'structured_content'
                    })
                    current_pos = heading_pos + len(heading)
        
        # This will remove newlines, create semantic chunks, and map to sections
        # If no source_sections found, it will auto-detect sections
        return self.chunk_text_with_sections(
            text, 
            headings, 
            source_sections if source_sections else None,
            use_llm_for_sections=use_llm_for_sections,
            professor_name=professor_name,
            llm_clean_chunks=llm_clean_chunks,
            llm_timeout=llm_timeout
        )


# Singleton instance
_chunker_instance = None

def get_text_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> TextChunker:
    """Get or create text chunker instance"""
    global _chunker_instance
    
    if _chunker_instance is None or \
       _chunker_instance.chunk_size != chunk_size or \
       _chunker_instance.chunk_overlap != chunk_overlap:
        _chunker_instance = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return _chunker_instance


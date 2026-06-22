"""
URL categorization utility
Categorizes URLs into different types for processing priority
"""
from urllib.parse import urlparse
import re
import requests
from typing import Optional, Tuple


def get_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""


def is_research_portal(url: str) -> bool:
    """
    Check if URL is a research/publication portal
    Returns True for: Google Scholar, ResearchGate, PubMed, arXiv, etc.
    """
    url_lower = url.lower()
    research_portals = [
        'scholar.google.com',
        'researchgate.net',
        'pubmed.ncbi.nlm.nih.gov',
        'arxiv.org',
        'ieee.org',
        'acm.org',
        'springer.com',
        'sciencedirect.com',
        'nature.com',
        'science.org',
        'plos.org',
        'biorxiv.org',
        'medrxiv.org',
        'dblp.org',
        'dblp.uni-trier.de',  # DBLP database
        'academia.edu',
        'orcid.org',
        'publons.com',
        'scopus.com',
        'webofscience.com',
        'jstor.org'
    ]
    return any(portal in url_lower for portal in research_portals)


def is_research_paper_link(url: str, link_text: str = "") -> bool:
    """
    Check if URL points to a research paper/publication
    Returns True if URL or link text suggests it's a research paper
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    
    # Check URL patterns
    paper_url_patterns = [
        '/article/',
        '/paper/',
        '/publication/',
        '/pub/',
        '/doi/',
        '/abs/',
        '/pdf/',
        '/eprint/',
        'arxiv.org/abs/',
        'pubmed.ncbi.nlm.nih.gov/',
        'doi.org/',
        'dx.doi.org/'
    ]
    if any(pattern in url_lower for pattern in paper_url_patterns):
        return True
    
    # Check link text keywords
    paper_keywords = [
        'paper', 'publication', 'article', 'research', 'study',
        'journal', 'conference', 'proceedings', 'abstract',
        'doi:', 'doi ', 'arxiv:', 'pubmed:', 'citation'
    ]
    if text_lower and any(keyword in text_lower for keyword in paper_keywords):
        return True
    
    # Check if it's from a research portal
    if is_research_portal(url):
        # If it's a specific paper URL (not just the profile), it's likely a paper
        if any(indicator in url_lower for indicator in ['/user/', '/profile/', '/author/']):
            return False  # Profile page, not a paper
        return True  # Likely a paper page
    
    return False


def is_browse_or_search_page(url: str, link_text: str = "") -> bool:
    """
    Check if URL is a browse/search/listing page (not a specific profile/page)
    Returns True for: browse pages, search results, directory listings, etc.
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    
    # URL patterns that indicate browse/search pages
    browse_patterns = [
        '/browse',
        '/browse?',
        '/search',
        '/search?',
        '/results',
        '/results?',
        '/listing',
        '/list',
        '/directory',
        '/profiles?',
        '/people?',
        '/faculty?',
        '/staff?',
        '?search=',
        '?q=',
        '?query=',
        '?keyword=',
        '?filter=',
        '?category=',
        '?tag=',
        '/tag/',
        '/category/',
        'view all',
        'see all',
        'more results'
    ]
    
    # Check URL patterns
    if any(pattern in url_lower for pattern in browse_patterns):
        return True
    
    # Check link text for browse/search indicators
    browse_keywords = [
        'browse', 'search', 'results', 'listing', 'directory',
        'view all', 'see all', 'more', 'all profiles', 'all people',
        'all faculty', 'all staff', 'find more', 'show more'
    ]
    if text_lower and any(keyword in text_lower for keyword in browse_keywords):
        return True
    
    return False


def categorize_url(url: str, link_text: str = "", base_domain: str = "") -> str:
    """
    Categorize URLs into different types:
    - document: PDF, Word, text files
    - research_publication: Research papers/publications (extract titles only)
    - research_portal: Research portals like Google Scholar (extract titles only)
    - personal_website: External personal websites
    - academic_profile: Google Scholar, ResearchGate, etc.
    - social_media: Twitter, LinkedIn, etc.
    - internal_page: Same domain pages
    - other: Unclassified
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    
    # Check for documents FIRST (before other checks) - use dynamic detection
    if is_document_url(url, link_text, use_dynamic=True):
        return 'document'
    
    # Check for research papers/publications (extract titles only)
    if is_research_paper_link(url, link_text):
        return 'research_publication'
    
    # Check for browse/search pages (skip these)
    if is_browse_or_search_page(url, link_text):
        return 'browse_page'
    
    # Check for research portals (extract titles only)
    if is_research_portal(url):
        return 'research_portal'
    
    # Check for academic profiles
    academic_domains = [
        'scholar.google.com',
        'researchgate.net',
        'orcid.org',
        'academia.edu',
        'publons.com',
        'dblp.org',
        'scopus.com'
    ]
    if any(domain in url_lower for domain in academic_domains):
        return 'academic_profile'
    
    # Check for personal website indicators
    personal_indicators = [
        'personal', 'homepage', 'website', 'home page',
        'www.', 'http://', 'https://'
    ]
    
    url_domain = get_domain(url)
    
    # External link (different domain) + personal indicators
    if base_domain and url_domain != base_domain.lower():
        if any(indicator in text_lower for indicator in personal_indicators):
            return 'personal_website'
        # External but not academic = likely personal website
        if url_domain and not any(domain in url_lower for domain in academic_domains):
            return 'personal_website'
    
    # Social media
    social_domains = [
        'twitter.com', 'x.com', 'linkedin.com', 
        'facebook.com', 'instagram.com'
    ]
    if any(domain in url_lower for domain in social_domains):
        return 'social_media'
    
    # Internal page
    if base_domain and url_domain == base_domain.lower():
        return 'internal_page'
    
    # Email links
    if url_lower.startswith('mailto:'):
        return 'email'
    
    return 'other'


def get_url_type(url: str) -> str:
    """Get basic URL type: document, external, internal, email"""
    url_lower = url.lower()
    
    # Documents
    document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
    if any(url_lower.endswith(ext) for ext in document_extensions):
        return 'document'
    
    # Email
    if url_lower.startswith('mailto:'):
        return 'email'
    
    # External vs Internal (requires base_domain to be accurate)
    return 'webpage'


def _check_content_type_header(url: str) -> Optional[Tuple[bool, str]]:
    """
    Check HTTP Content-Type header to determine if URL is a document
    Returns: (is_document: bool, content_type: str) or None if check fails
    """
    try:
        # Disable SSL verification for self-signed certificates
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Make HEAD request (lightweight, no body download)
        response = requests.head(url, timeout=10, allow_redirects=True, verify=False)
        content_type = response.headers.get('content-type', '').lower()
        
        if not content_type:
            return None
        
        # Document MIME types
        document_mime_types = [
            'application/pdf',
            'application/msword',  # .doc
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
            'application/vnd.ms-word',  # .doc
            'application/rtf',
            'text/plain',
            'application/octet-stream',  # Sometimes used for documents
        ]
        
        # Check if content-type indicates a document
        is_document = any(doc_type in content_type for doc_type in document_mime_types)
        
        # Also check for HTML/text - if it's HTML, it's likely a webpage
        if 'text/html' in content_type or 'application/xhtml' in content_type:
            return (False, content_type)
        
        return (is_document, content_type)
    except Exception as e:
        # If HEAD fails, return None (will try other methods)
        return None


def _check_file_magic_numbers(url: str) -> Optional[bool]:
    """
    Download first few bytes and check file magic numbers (file signatures)
    Returns: True if document, False if not, None if check fails
    """
    try:
        # Disable SSL verification for self-signed certificates
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Download only first 512 bytes (enough for magic number detection)
        response = requests.get(url, timeout=10, stream=True, headers={'Range': 'bytes=0-511'}, verify=False)
        response.raise_for_status()
        
        # Read first bytes
        first_bytes = response.content[:512]
        
        if len(first_bytes) < 4:
            return None
        
        # PDF magic number: %PDF
        if first_bytes[:4] == b'%PDF':
            return True
        
        # DOCX magic number: PK\x03\x04 (ZIP format)
        if first_bytes[:4] == b'PK\x03\x04':
            # Check if it's a DOCX by looking for word/ in the ZIP structure
            if b'word/' in first_bytes[:100]:
                return True
        
        # DOC (OLE2 format): D0 CF 11 E0 A1 B1 1A E1
        if first_bytes[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return True
        
        # RTF magic number: {\rtf
        if first_bytes[:5] == b'{\\rtf':
            return True
        
        # HTML indicators (not a document)
        if b'<!DOCTYPE' in first_bytes[:100] or b'<html' in first_bytes[:100].lower():
            return False
        
        # If we can't determine, return None
        return None
    except Exception as e:
        return None


def is_cv_link(url: str, link_text: str = "") -> bool:
    """
    Check if URL is a CV/Curriculum Vitae/Resume link
    Returns True if URL or link text strongly suggests it's a CV/resume
    
    Args:
        url: URL to check
        link_text: Link text (optional, for keyword detection)
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    
    # CV-specific keywords (strong indicators)
    cv_keywords = [
        'cv', 'curriculum vitae', 'resume', 'resumé', 'curriculum', 'vitae',
        'c.v.', 'cv.', 'cv-', 'cv_', 'resume.', 'resume-', 'resume_'
    ]
    
    # Check link text for CV keywords (strongest indicator)
    if text_lower:
        if any(keyword in text_lower for keyword in cv_keywords):
            return True
    
    # Check URL for CV keywords
    if any(keyword in url_lower for keyword in cv_keywords):
        return True
    
    # Check if it's a document with CV-like filename patterns
    document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
    if any(url_lower.endswith(ext) for ext in document_extensions):
        # Check if filename contains CV indicators
        # Extract filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.lower()
        filename = path.split('/')[-1] if path else ''
        
        # Check filename for CV patterns
        cv_filename_patterns = ['cv', 'resume', 'curriculum', 'vitae']
        if any(pattern in filename for pattern in cv_filename_patterns):
            return True
    
    return False


def is_personal_website_link(url: str, link_text: str = "", base_domain: str = "") -> bool:
    """
    Check if URL is a personal website link
    Returns True if URL appears to be a personal website (not academic profile, social media, etc.)
    
    Args:
        url: URL to check
        link_text: Link text (optional, for keyword detection)
        base_domain: Base domain of the profile page (to determine if external)
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    url_domain = get_domain(url)
    
    # Skip if it's not an external link (same domain as profile)
    if base_domain and url_domain.lower() == base_domain.lower():
        return False
    
    # Skip email links
    if url_lower.startswith('mailto:'):
        return False
    
    # Skip research portals and academic profiles (use comprehensive function)
    if is_research_portal(url):
        return False
    
    # Skip research paper/publication links
    if is_research_paper_link(url, link_text):
        return False
    
    # Skip browse/search pages
    if is_browse_or_search_page(url, link_text):
        return False
    
    # Skip research databases and bibliographic services
    research_databases = [
        'dblp.org', 'dblp.uni-trier.de', 'aclanthology.org', 'anthology.aclweb.org',
        'semanticscholar.org', 'citeseerx.ist.psu.edu', 'citeseer.ist.psu.edu',
        'microsoft.com/academic', 'aminer.org', 'connectedpapers.com'
    ]
    if any(db in url_lower for db in research_databases):
        return False
    
    # Skip funding agency websites
    funding_agencies = [
        'nsf.gov', 'nih.gov', 'darpa.mil', 'nsa.gov', 'doe.gov',
        'erc.europa.eu', 'wellcome.org', 'gatesfoundation.org',
        'awardsearch', 'grants.gov', 'research.gov'
    ]
    if any(agency in url_lower for agency in funding_agencies):
        return False
    
    # Skip conference/journal organization websites
    research_orgs = [
        'sigmod.org', 'sigmodrecord.org', 'vldb.org', 'icde.org',
        'computer.org', 'ieee.org', 'acm.org', 'usenix.org',
        'aaai.org', 'ijcai.org', 'neurips.cc', 'icml.cc',
        'iclr.cc', 'aclweb.org', 'emnlp.org', 'naacl.org'
    ]
    if any(org in url_lower for org in research_orgs):
        return False
    
    # Skip lab/research group pages (common patterns)
    lab_patterns = [
        '/lab/', '/labs/', '/group/', '/groups/', '/research-group/',
        '/research-group/', '/team/', '/teams/', '/seminar/', '/seminars/',
        '/workshop/', '/workshops/', '/indexlab', '/datalab', '/researchlab'
    ]
    if any(pattern in url_lower for pattern in lab_patterns):
        return False
    
    # Skip research group indicators in domain/subdomain
    # Check for GitHub Pages with research group indicators
    if '.github.io' in url_lower:
        # Check if subdomain contains research group keywords
        research_group_keywords = ['group', 'lab', 'research', 'database', 'datalab', 'indexlab', 'seminar', 'workshop']
        # Extract subdomain from URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            if '.github.io' in hostname:
                subdomain = hostname.replace('.github.io', '')
                if any(keyword in subdomain for keyword in research_group_keywords):
                    return False
        except:
            pass
    
    # Check domain/subdomain for research group indicators
    research_group_domains = ['research', 'lab', 'group', 'datalab', 'indexlab']
    if any(indicator in url_domain.lower() for indicator in research_group_domains):
        # But allow if it's clearly a personal website (check link text)
        if not text_lower or not any(indicator in text_lower for indicator in ['personal', 'homepage', 'website', 'my site']):
            return False
    
    # Skip social media and professional networks (EXCLUDE ALL)
    social_domains = [
        'twitter.com', 'x.com', 'linkedin.com', 
        'facebook.com', 'instagram.com', 'youtube.com', 'github.com',
        'pinterest.com', 'tumblr.com', 'reddit.com', 'medium.com',
        'snapchat.com', 'tiktok.com', 'whatsapp.com', 'telegram.org',
        'discord.com', 'slack.com', 'mastodon.social', 'threads.net',
        'bluesky.social', 'bsky.app',
        'quora.com', 'stackoverflow.com', 'stackexchange.com'
    ]
    if any(domain in url_lower for domain in social_domains):
        return False
    
    # Skip journal/conference websites (common patterns)
    journal_indicators = [
        'journal', 'conference', 'proceedings', 'symposium', 'workshop',
        'ieee', 'acm', 'springer', 'elsevier', 'wiley', 'taylor', 'francis',
        'sciencedirect', 'nature', 'science', 'cell', 'lancet', 'nejm',
        'record.org', 'bulletin', 'debull'  # SIGMOD Record, IEEE Data Engineering Bulletin, etc.
    ]
    # Check if domain or URL contains journal indicators
    if any(indicator in url_lower for indicator in journal_indicators):
        # But allow if it's clearly a personal website (e.g., "myjournal.com" would be caught, but that's rare)
        # More importantly, check if link text suggests it's a personal site
        if text_lower and not any(indicator in text_lower for indicator in ['personal', 'homepage', 'website', 'my site']):
            return False
    
    # Skip institutional/university department pages (common patterns)
    institutional_patterns = [
        '/department/', '/dept/', '/faculty/', '/staff/', '/people/',
        '/directory/', '/profiles/', '/faculty-staff/', '/about/faculty',
        '.edu/department', '.edu/faculty', '.edu/people'
    ]
    if any(pattern in url_lower for pattern in institutional_patterns):
        return False
    
    # Skip documents (CVs are handled separately)
    if is_document_url(url, link_text, use_dynamic=False):
        return False
    
    # Skip if link text suggests it's a research/publication link
    research_text_indicators = [
        'publication', 'paper', 'article', 'research', 'study', 'journal',
        'conference', 'proceedings', 'abstract', 'citation', 'doi', 'arxiv',
        'pubmed', 'scholar', 'researchgate'
    ]
    if text_lower and any(indicator in text_lower for indicator in research_text_indicators):
        return False
    
    # Check for personal website indicators in link text (positive signal)
    personal_indicators = [
        'personal', 'homepage', 'website', 'home page', 'homepage',
        'my website', 'my site', 'personal site', 'personal website'
    ]
    if text_lower and any(indicator in text_lower for indicator in personal_indicators):
        return True
    
    # If it's an external link (different domain) and passed all exclusion checks,
    # it's likely a personal website
    if base_domain and url_domain and url_domain.lower() != base_domain.lower():
        # Additional safety check for .edu domains (university websites)
        if url_domain.endswith('.edu'):
            # Personal pages on university servers typically have these patterns:
            personal_edu_patterns = ['/~', '/users/', '/user/', '/home/']
            has_personal_pattern = any(pattern in url_lower for pattern in personal_edu_patterns)
            
            # If link text suggests personal website, allow it
            has_personal_text = text_lower and any(indicator in text_lower for indicator in personal_indicators)
            
            # If it has personal page pattern or personal text, allow it
            if has_personal_pattern or has_personal_text:
                return True
            else:
                # Likely an institutional/department page, skip it
                return False
        
        # For non-.edu domains, if it passed all exclusion checks, it's likely a personal website
        return True
    
    return False


def is_document_url(url: str, link_text: str = "", use_dynamic: bool = True) -> bool:
    """
    Dynamically check if URL points to a document file
    
    Detection methods (in order):
    1. Link text keywords (fast, no network request)
    2. URL extension check (fast, no network request)
    3. HTTP Content-Type header (HEAD request)
    4. File magic numbers (first bytes)
    
    Args:
        url: URL to check
        link_text: Link text (optional, for keyword detection)
        use_dynamic: If True, use HTTP headers/magic numbers. If False, use fast checks only.
    """
    url_lower = url.lower()
    text_lower = link_text.lower() if link_text else ""
    
    # Method 1: Check link text for document keywords (fastest, no network)
    document_keywords = [
        'cv', 'curriculum vitae', 'resume', 'resumé',
        'pdf', 'document', 'download', 'file', 'attachment'
    ]
    if text_lower and any(keyword in text_lower for keyword in document_keywords):
        return True
    
    # Method 2: Check URL extension (fast, no network)
    document_extensions = ['.pdf', '.doc', '.docx', '.txt', '.rtf']
    if any(url_lower.endswith(ext) for ext in document_extensions):
        return True
    
    # If use_dynamic=False, stop here (fast checks only)
    if not use_dynamic:
        return False
    
    # Method 3: Check Content-Type header (lightweight network request)
    content_type_result = _check_content_type_header(url)
    if content_type_result is not None:
        is_doc, content_type = content_type_result
        if is_doc:
            return True
        # If explicitly HTML, it's not a document
        if 'text/html' in content_type:
            return False
    
    # Method 4: Check file magic numbers (download first bytes)
    magic_result = _check_file_magic_numbers(url)
    if magic_result is not None:
        return magic_result
    
    # Method 5: Check for document hosting services in URL (fallback)
    document_hosting_patterns = [
        'sharepoint.com',
        'onedrive.live.com',
        'drive.google.com',
        'dropbox.com',
        'box.com',
    ]
    # Only if link text suggests it's a document
    if text_lower and any(keyword in text_lower for keyword in ['cv', 'curriculum vitae', 'resume', 'document', 'file']):
        if any(pattern in url_lower for pattern in document_hosting_patterns):
            return True
    
    # Default: assume it's not a document if we can't determine
    return False




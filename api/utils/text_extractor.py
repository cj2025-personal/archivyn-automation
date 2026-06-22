"""
Text extraction utilities for web pages
"""
from bs4 import BeautifulSoup
import re


def extract_structured_text(html_content: str) -> dict:
    """
    Extract structured text content from HTML
    Returns dictionary with different sections
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove ALL iframes and embedded social media widgets FIRST (before any other processing)
    for iframe in soup.find_all(['iframe', 'embed', 'object']):
        src = iframe.get('src') or iframe.get('data-src') or ''
        src_lower = src.lower() if src else ''
        # Remove iframes/embeds pointing to social media
        if any(domain in src_lower for domain in ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'x.com', 'youtube.com']):
            iframe.decompose()
        # Remove all iframes/embeds to be safe (they often contain widgets)
        elif iframe.name in ['iframe', 'embed', 'object']:
            iframe.decompose()
    
    # Remove social media widget containers
    social_widget_classes = ['fb-', 'facebook', 'twitter', 'instagram', 'linkedin', 'social-widget', 'social-media']
    for widget in soup.find_all(class_=lambda x: x and any(social in str(x).lower() for social in social_widget_classes)):
        widget.decompose()
    for widget in soup.find_all(id=lambda x: x and any(social in str(x).lower() for social in social_widget_classes)):
        widget.decompose()
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    result = {
        'full_text': '',
        'bio': '',
        'research_interests': '',
        'education': '',
        'publications': '',
        'contact_info': ''
    }
    
    # Try to find common sections
    # Bio/About section
    bio_selectors = [
        'div.bio', 'div.about', 'div#about', 'section.about',
        'div.profile', 'div.description', 'p.bio'
    ]
    for selector in bio_selectors:
        element = soup.select_one(selector)
        if element:
            result['bio'] = element.get_text(strip=True, separator=' ')
            break
    
    # Research interests
    research_selectors = [
        'div.research', 'div.interests', 'div.research-interests',
        'section.research', 'ul.research-interests'
    ]
    for selector in research_selectors:
        element = soup.select_one(selector)
        if element:
            result['research_interests'] = element.get_text(strip=True, separator=' ')
            break
    
    # Education
    education_selectors = [
        'div.education', 'section.education', 'div#education'
    ]
    for selector in education_selectors:
        element = soup.select_one(selector)
        if element:
            result['education'] = element.get_text(strip=True, separator=' ')
            break
    
    # Publications
    publication_selectors = [
        'div.publications', 'section.publications', 'div#publications',
        'ul.publications', 'ol.publications'
    ]
    for selector in publication_selectors:
        element = soup.select_one(selector)
        if element:
            result['publications'] = element.get_text(strip=True, separator=' ')
            break
    
    # Contact info (email, phone)
    text = soup.get_text()
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        result['contact_info'] = ', '.join(emails)
    
    # Full text (fallback)
    full_text = soup.get_text(separator=' ', strip=True)
    
    # Check if extracted text is primarily social media content
    # But be smarter: check if content is PRIMARILY social media vs just having social links
    text_lower = full_text.lower()
    social_media_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'log in', 'forgot password', 'create account', 'followers', 'following', 'posts', 'reels', 'photos']
    profile_content_keywords = ['education', 'research', 'publications', 'teaching', 'biography', 'bio', 'professor', 'ph.d', 'phd', 'university', 'department', 'curriculum vitae', 'cv', 'awards', 'honors', 'courses', 'interests']
    
    social_media_count = sum(1 for keyword in social_media_keywords if keyword in text_lower)
    profile_content_count = sum(1 for keyword in profile_content_keywords if keyword in text_lower)
    word_count = len(full_text.split())
    
    # Only mark as social media if:
    # 1. Has 3+ social media keywords AND
    # 2. Has very few profile content keywords (< 2) AND
    # 3. Text is relatively short (< 200 words) OR social media keywords are very frequent
    is_social_media = social_media_count >= 3 and profile_content_count < 2 and (word_count < 200 or social_media_count >= 5)
    
    if is_social_media:
        result['full_text'] = ''
    else:
        result['full_text'] = full_text
    
    return result


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = text.strip()
    return text




"""
HTML Cleaner Service - Convert HTML tags to markdown formatting

Handles:
- HTML tag removal and conversion
- Special characters and entities
- Text formatting (bold, italic, underline)
- Links and images
- Lists and tables
- Code blocks
"""

import re
import html
from typing import Optional, Dict, List
from html.parser import HTMLParser


class HTMLCleanerService:
    """Service to clean and convert HTML tags to markdown format."""
    
    @staticmethod
    def sanitize_response(response_text: str) -> str:
        """
        Sanitize LLM response by removing/converting HTML tags.
        
        Args:
            response_text: Raw response text with HTML tags
            
        Returns:
            Cleaned markdown-formatted text
        """
        if not response_text:
            return response_text
        
        text = HTMLCleanerService.html_to_markdown(response_text)
        return text
    
    @staticmethod
    def html_to_markdown(html_text: str) -> str:
        """
        Convert HTML to markdown for better readability.
        
        Handles:
        - Line breaks: <br> → newline
        - Paragraphs: <p> tags → newline
        - Bold: <strong>, <b> → **text**
        - Italic: <em>, <i> → *text*
        - Underline: <u> → __text__
        - Headers: <h1-4> → # ## ### ####
        - Links: <a href="url"> → [text](url)
        - Images: <img> → ![alt](src)
        - Lists: <ul>, <ol>, <li> → bullet points
        - Code: <code> → `text`
        - Pre: <pre> → code block
        
        Args:
            html_text: HTML formatted text
            
        Returns:
            Markdown formatted text
        """
        if not html_text:
            return html_text
        
        text = html_text
        
        # 1. Remove script and style tags completely
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. Handle line breaks and paragraphs
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<br\s*/?\s*>\s*', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>\s*<p[^>]*>', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        
        # 3. Handle text formatting tags (order matters - do nested tags first)
        # Strong/Bold
        text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Emphasis/Italic
        text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Underline
        text = re.sub(r'<u[^>]*>(.*?)</u>', r'__\1__', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Strikethrough
        text = re.sub(r'<s[^>]*>(.*?)</s>', r'~~\1~~', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<del[^>]*>(.*?)</del>', r'~~\1~~', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 4. Handle headers
        text = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h5[^>]*>(.*?)</h5>', r'##### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<h6[^>]*>(.*?)</h6>', r'###### \1\n', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 5. Handle links
        text = re.sub(
            r'<a\s+[^>]*href=(["\'])([^"\']*)\1[^>]*>(.*?)</a>',
            r'[\3](\2)',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # Alternative link format without quotes
        text = re.sub(
            r'<a\s+[^>]*href=([^\s>]+)[^>]*>(.*?)</a>',
            r'[\2](\1)',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # 6. Handle images
        text = re.sub(
            r'<img[^>]*src=(["\'])([^"\']*)\1[^>]*alt=(["\'])([^"\']*)\3[^>]*>',
            r'![\4](\2)',
            text,
            flags=re.IGNORECASE
        )
        # Alternative format
        text = re.sub(
            r'<img[^>]*alt=(["\'])([^"\']*)\1[^>]*src=(["\'])([^"\']*)\3[^>]*>',
            r'![\2](\4)',
            text,
            flags=re.IGNORECASE
        )
        # Fallback - just src
        text = re.sub(
            r'<img[^>]*src=(["\'])([^"\']*)\1[^>]*>',
            r'![\2](\2)',
            text,
            flags=re.IGNORECASE
        )
        
        # 7. Handle code tags
        text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 8. Handle pre/code blocks
        text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'```\n\1\n```', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 9. Handle lists
        # Unordered lists
        text = re.sub(r'<ul[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</ul>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<ol[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</ol>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<li[^>]*>', '• ', text, flags=re.IGNORECASE)
        text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
        
        # 10. Handle horizontal rule
        text = re.sub(r'<hr\s*/?>', '\n---\n', text, flags=re.IGNORECASE)
        
        # 11. Handle blockquotes
        text = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', r'> \1', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 12. Handle divs and sections (just remove tags)
        text = re.sub(r'<div[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<section[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</section>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<article[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</article>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<main[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</main>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<header[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</header>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</footer>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</nav>', '\n', text, flags=re.IGNORECASE)
        
        # 13. Handle spans and other formatting tags
        text = re.sub(r'<span[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</span>', '', text, flags=re.IGNORECASE)
        
        # 14. Handle tables (basic conversion)
        text = re.sub(r'<table[^>]*>', '| ', text, flags=re.IGNORECASE)
        text = re.sub(r'</table>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<tr[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<th[^>]*>', '| ', text, flags=re.IGNORECASE)
        text = re.sub(r'</th>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<td[^>]*>', '| ', text, flags=re.IGNORECASE)
        text = re.sub(r'</td>', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'<thead[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</thead>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<tbody[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</tbody>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<tfoot[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</tfoot>', '', text, flags=re.IGNORECASE)
        
        # 15. Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # 16. Decode HTML entities
        text = html.unescape(text)
        
        # 17. Clean up whitespace
        # Remove multiple consecutive newlines (more than 2)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def strip_html_tags(html_text: str) -> str:
        """
        Strip all HTML tags, keeping only text content.
        
        Args:
            html_text: HTML text to clean
            
        Returns:
            Plain text without HTML tags
        """
        # Remove tags
        text = re.sub(r'<[^>]+>', '', html_text)
        
        # Decode entities
        text = html.unescape(text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def is_html_content(text: str) -> bool:
        """
        Check if text contains HTML tags.
        
        Args:
            text: Text to check
            
        Returns:
            True if HTML tags detected
        """
        return bool(re.search(r'<[^>]+>', text))
    
    @staticmethod
    def extract_text_from_html(html_text: str) -> str:
        """
        Extract plain text from HTML while preserving structure.
        
        Args:
            html_text: HTML text
            
        Returns:
            Plain text with markdown formatting
        """
        return HTMLCleanerService.html_to_markdown(html_text)
    
    @staticmethod
    def clean_text_content(content: str) -> str:
        """
        Clean text content by:
        1. Removing HTML tags
        2. Decoding entities
        3. Removing extra whitespace
        4. Normalizing line breaks
        
        Args:
            content: Raw content
            
        Returns:
            Cleaned content
        """
        if not content:
            return content
        
        # First pass: convert HTML to markdown
        text = HTMLCleanerService.html_to_markdown(content)
        
        # Ensure consistent line breaks
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        
        return text
    
    @staticmethod
    def get_html_tag_info(html_text: str) -> Dict[str, int]:
        """
        Get information about HTML tags in text.
        
        Args:
            html_text: HTML text
            
        Returns:
            Dictionary with tag counts and types
        """
        tags = re.findall(r'<([a-z0-9]+)[^>]*>', html_text, re.IGNORECASE)
        
        tag_info = {}
        for tag in tags:
            tag_lower = tag.lower()
            tag_info[tag_lower] = tag_info.get(tag_lower, 0) + 1
        
        return tag_info


class HTMLParser_Custom(HTMLParser):
    """Custom HTML parser for extracting structured content."""
    
    def __init__(self):
        super().__init__()
        self.text_content = []
        self.links = []
        self.images = []
        self.current_tag = None
        self.skip_content = False
    
    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        if tag in ['script', 'style']:
            self.skip_content = True
        
        self.current_tag = tag
        
        # Extract links
        if tag == 'a':
            for attr, value in attrs:
                if attr == 'href':
                    self.links.append(value)
        
        # Extract images
        if tag == 'img':
            img_data = {'src': '', 'alt': ''}
            for attr, value in attrs:
                if attr == 'src':
                    img_data['src'] = value
                elif attr == 'alt':
                    img_data['alt'] = value
            if img_data['src']:
                self.images.append(img_data)
    
    def handle_endtag(self, tag):
        """Handle closing tags."""
        if tag in ['script', 'style']:
            self.skip_content = False
        
        self.current_tag = None
    
    def handle_data(self, data):
        """Handle text content."""
        if not self.skip_content and data.strip():
            self.text_content.append(data.strip())
    
    def get_text(self) -> str:
        """Get extracted text content."""
        return ' '.join(self.text_content)
    
    def get_links(self) -> List[str]:
        """Get all links found."""
        return self.links
    
    def get_images(self) -> List[Dict]:
        """Get all images found."""
        return self.images


def clean_response_html(response: str) -> str:
    """
    Convenience function to clean HTML from response.
    
    Args:
        response: Raw response text
        
    Returns:
        Cleaned response
    """
    return HTMLCleanerService.sanitize_response(response)


def is_response_contains_html(response: str) -> bool:
    """
    Check if response contains HTML tags.
    
    Args:
        response: Response text
        
    Returns:
        True if HTML found
    """
    return HTMLCleanerService.is_html_content(response)

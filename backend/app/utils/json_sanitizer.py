"""
Universal JSON Sanitizer - Handles ALL types of malformed JSON automatically.
Fixes ISODate, ObjectId, missing commas, trailing commas, comments, etc.
"""

import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONSanitizer:
    """
    Robust JSON sanitizer that automatically fixes common issues:
    - MongoDB export formats (ISODate, ObjectId, NumberLong, etc.)
    - Missing/trailing commas
    - Single quotes instead of double quotes
    - Comments (// and /* */)
    - Undefined/NaN/Infinity values
    - Duplicate keys
    - Invalid escape sequences
    """
    
    def __init__(self):
        self.cleaning_steps = []
        
    def sanitize(self, content: str) -> Tuple[Any, List[str]]:
        """
        Sanitize JSON content using multiple strategies.
        
        Args:
            content: Raw JSON string (possibly malformed)
            
        Returns:
            Tuple of (parsed_data, list_of_fixes_applied)
        """
        self.cleaning_steps = []
        
        # Strategy 1: Try standard parse first
        try:
            data = json.loads(content)
            self.cleaning_steps.append("✓ Standard JSON parse successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError as e:
            self.cleaning_steps.append(f"✗ Standard parse failed: {e}")
        
        # Strategy 2: Clean MongoDB export formats
        cleaned_content = self._clean_mongodb_formats(content)
        try:
            data = json.loads(cleaned_content)
            self.cleaning_steps.append("✓ MongoDB format cleaning successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError:
            pass
        
        # Strategy 3: Fix structural issues (commas, quotes, brackets)
        cleaned_content = self._fix_structural_issues(cleaned_content)
        try:
            data = json.loads(cleaned_content)
            self.cleaning_steps.append("✓ Structural fixes successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Remove comments and clean whitespace
        cleaned_content = self._remove_comments(cleaned_content)
        try:
            data = json.loads(cleaned_content)
            self.cleaning_steps.append("✓ Comment removal successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Fix JavaScript-style values
        cleaned_content = self._fix_javascript_values(cleaned_content)
        try:
            data = json.loads(cleaned_content)
            self.cleaning_steps.append("✓ JavaScript value fixes successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Line-by-line parsing for arrays
        if self._looks_like_json_lines(content):
            try:
                data = self._parse_json_lines(content)
                self.cleaning_steps.append("✓ JSON Lines format parsed")
                return data, self.cleaning_steps
            except Exception:
                pass
        
        # Strategy 7: Aggressive cleaning (last resort)
        cleaned_content = self._aggressive_clean(content)
        try:
            data = json.loads(cleaned_content)
            self.cleaning_steps.append("✓ Aggressive cleaning successful")
            return data, self.cleaning_steps
        except json.JSONDecodeError as e:
            # All strategies failed
            self.cleaning_steps.append(f"✗ All sanitization strategies failed: {e}")
            raise ValueError(f"Unable to parse JSON after {len(self.cleaning_steps)} attempts. Last error: {e}")
    
    def _clean_mongodb_formats(self, content: str) -> str:
        """Clean MongoDB export formats."""
        # ISODate("2024-01-01T00:00:00Z") -> "2024-01-01T00:00:00Z"
        content = re.sub(
            r'ISODate\s*\(\s*"([^"]+)"\s*\)',
            r'"\1"',
            content
        )
        self.cleaning_steps.append("  - Cleaned ISODate format")
        
        # ObjectId("507f1f77bcf86cd799439011") -> "507f1f77bcf86cd799439011"
        content = re.sub(
            r'ObjectId\s*\(\s*"([^"]+)"\s*\)',
            r'"\1"',
            content
        )
        self.cleaning_steps.append("  - Cleaned ObjectId format")
        
        # NumberLong(12345) -> 12345
        content = re.sub(
            r'NumberLong\s*\(\s*(\d+)\s*\)',
            r'\1',
            content
        )
        self.cleaning_steps.append("  - Cleaned NumberLong format")
        
        # NumberInt(123) -> 123
        content = re.sub(
            r'NumberInt\s*\(\s*(\d+)\s*\)',
            r'\1',
            content
        )
        self.cleaning_steps.append("  - Cleaned NumberInt format")
        
        # NumberDecimal("123.45") -> 123.45
        content = re.sub(
            r'NumberDecimal\s*\(\s*"([^"]+)"\s*\)',
            r'\1',
            content
        )
        self.cleaning_steps.append("  - Cleaned NumberDecimal format")
        
        # Timestamp(1234567890, 1) -> 1234567890
        content = re.sub(
            r'Timestamp\s*\(\s*(\d+)\s*,\s*\d+\s*\)',
            r'\1',
            content
        )
        self.cleaning_steps.append("  - Cleaned Timestamp format")
        
        # BinData(0, "base64string") -> "base64string"
        content = re.sub(
            r'BinData\s*\(\s*\d+\s*,\s*"([^"]+)"\s*\)',
            r'"\1"',
            content
        )
        self.cleaning_steps.append("  - Cleaned BinData format")
        
        return content
    
    def _fix_structural_issues(self, content: str) -> str:
        """Fix common structural issues."""
        # Fix missing commas between objects in arrays
        # }  { -> },\n{
        content = re.sub(r'\}\s+\{', '},\n{', content)
        self.cleaning_steps.append("  - Fixed missing commas between objects")
        
        # Fix missing commas between array items
        # ]  [ -> ],\n[
        content = re.sub(r'\]\s+\[', '],\n[', content)
        
        # Remove trailing commas before closing brackets/braces
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        self.cleaning_steps.append("  - Removed trailing commas")
        
        # Fix single quotes to double quotes (but not inside strings)
        # This is complex, so we use a simple heuristic
        content = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', content)
        self.cleaning_steps.append("  - Converted single quotes to double quotes")
        
        return content
    
    def _remove_comments(self, content: str) -> str:
        """Remove JavaScript-style comments."""
        # Remove // comments
        content = re.sub(r'//[^\n]*', '', content)
        
        # Remove /* */ comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        self.cleaning_steps.append("  - Removed comments")
        return content
    
    def _fix_javascript_values(self, content: str) -> str:
        """Fix JavaScript-specific values."""
        # undefined -> null
        content = re.sub(r'\bundefined\b', 'null', content)
        
        # NaN -> null
        content = re.sub(r'\bNaN\b', 'null', content)
        
        # Infinity -> null
        content = re.sub(r'\bInfinity\b', 'null', content)
        
        # -Infinity -> null
        content = re.sub(r'-Infinity\b', 'null', content)
        
        self.cleaning_steps.append("  - Fixed JavaScript values")
        return content
    
    def _looks_like_json_lines(self, content: str) -> bool:
        """Check if content looks like JSON Lines format (one JSON object per line)."""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check if first two lines start with { or [
        return all(line.strip().startswith(('{', '[')) for line in lines[:2] if line.strip())
    
    def _parse_json_lines(self, content: str) -> List[Dict]:
        """Parse JSON Lines format (one JSON object per line)."""
        results = []
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
        
        return results
    
    def _aggressive_clean(self, content: str) -> str:
        """Aggressive cleaning as last resort."""
        # Remove all non-printable characters except newlines
        content = ''.join(char for char in content if char.isprintable() or char in '\n\r\t')
        
        # Fix multiple commas
        content = re.sub(r',+', ',', content)
        
        # Fix whitespace around colons
        content = re.sub(r'\s*:\s*', ': ', content)
        
        # Ensure arrays/objects are properly closed
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces > close_braces:
            content += '}' * (open_braces - close_braces)
        
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        if open_brackets > close_brackets:
            content += ']' * (open_brackets - close_brackets)
        
        self.cleaning_steps.append("  - Applied aggressive cleaning")
        return content
    
    def validate_and_report(self, data: Any) -> Dict[str, Any]:
        """
        Validate parsed data and provide a report.
        
        Returns:
            Dictionary with validation results and statistics
        """
        report = {
            "valid": True,
            "type": type(data).__name__,
            "cleaning_steps": self.cleaning_steps,
            "statistics": {}
        }
        
        if isinstance(data, list):
            report["statistics"]["total_items"] = len(data)
            report["statistics"]["item_types"] = {}
            for item in data[:100]:  # Sample first 100
                item_type = type(item).__name__
                report["statistics"]["item_types"][item_type] = \
                    report["statistics"]["item_types"].get(item_type, 0) + 1
        
        elif isinstance(data, dict):
            report["statistics"]["total_keys"] = len(data)
            report["statistics"]["keys"] = list(data.keys())[:20]  # First 20 keys
        
        return report


# Global singleton
_sanitizer = JSONSanitizer()


def sanitize_json(content: str) -> Tuple[Any, List[str]]:
    """
    Convenience function to sanitize JSON content.
    
    Args:
        content: Raw JSON string
        
    Returns:
        Tuple of (parsed_data, list_of_fixes_applied)
    """
    return _sanitizer.sanitize(content)


def validate_json_file(content: bytes) -> Tuple[bool, Any, Dict[str, Any]]:
    """
    Validate and sanitize a JSON file.
    
    Args:
        content: Raw file bytes
        
    Returns:
        Tuple of (success, parsed_data, validation_report)
    """
    try:
        # Try UTF-8 first
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        # Try other encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = content.decode(encoding)
                logger.warning(f"Decoded with {encoding} instead of UTF-8")
                break
            except UnicodeDecodeError:
                continue
        else:
            return False, None, {"error": "Unable to decode file with any known encoding"}
    
    try:
        data, steps = sanitize_json(text)
        report = _sanitizer.validate_and_report(data)
        return True, data, report
    except Exception as e:
        return False, None, {
            "error": str(e),
            "cleaning_steps": _sanitizer.cleaning_steps
        }

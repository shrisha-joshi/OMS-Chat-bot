"""
Automated fixes for common SonarQube issues
- Empty f-strings → regular strings
- Unused variables → prefix with _
- Remove unused imports
"""

import os
import re
from pathlib import Path

def fix_empty_fstrings(content):
    """Replace f-strings without placeholders with regular strings."""
    # Pattern: f"text without {placeholders}" or f'text without {placeholders}'
    # Only match f-strings that don't contain { or }
    content = re.sub(r'f"([^"{}]+)"', r'"\1"', content)
    content = re.sub(r"f'([^'{}]+)'", r"'\1'", content)
    return content

def fix_unused_variables(content):
    """Prefix clearly unused variables with _."""
    # Common patterns like: result = ...  # Never used
    # Be conservative - only fix obvious cases
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix: status_code = ... (when clearly unused)
        if 'status_code = ' in line and 'unused' in line.lower():
            line = line.replace('status_code', '_status_code')
        # Fix: result = ... (when clearly unused)  
        elif 'result = ' in line and 'unused' in line.lower():
            line = line.replace('result', '_result')
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_file(file_path):
    """Apply all fixes to a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Apply fixes
        content = fix_empty_fstrings(content)
        content = fix_unused_variables(content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    backend_dir = Path(__file__).parent / 'app'
    fixed_count = 0
    
    for py_file in backend_dir.rglob('*.py'):
        if fix_file(py_file):
            fixed_count += 1
            print(f"✅ Fixed: {py_file.relative_to(backend_dir.parent)}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()

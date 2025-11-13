# pylint: disable=all
# type: ignore
# pylint: disable=duplicate-code,line-too-long
"""
Automated fix script for datetime.now(timezone.utc) deprecation
Replaces all instances with datetime.now(timezone.utc)
"""

import os
import re
from pathlib import Path

def fix_datetime_utcnow(file_path):
    """Fix datetime.now(timezone.utc) calls in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Check if timezone is imported
    needs_timezone_import = 'datetime.now(timezone.utc)' in content or 'datetime.datetime.now(datetime.timezone.utc)' in content
    has_timezone_import = 'from datetime import' in content and 'timezone' in content
    
    # Add timezone import if needed
    if needs_timezone_import and not has_timezone_import:
        # Find datetime import line
        if 'from datetime import datetime' in content:
            content = content.replace(
                'from datetime import datetime',
                'from datetime import datetime, timezone'
            )
        elif 'import datetime' in content:
            # Already imports datetime module, no change needed for import
            pass
    
    # Replace datetime.now(timezone.utc) with datetime.now(timezone.utc)
    content = re.sub(
        r'datetime\.datetime\.utcnow\(\)',
        'datetime.datetime.now(datetime.timezone.utc)',
        content
    )
    content = re.sub(
        r'datetime\.utcnow\(\)',
        'datetime.now(timezone.utc)',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    backend_dir = Path(__file__).parent
    fixed_files = []
    
    # Find all Python files
    for py_file in backend_dir.rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        try:
            if fix_datetime_utcnow(py_file):
                fixed_files.append(py_file)
                print(f"✅ Fixed: {py_file.relative_to(backend_dir)}")
        except Exception as e:
            print(f"❌ Error in {py_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total files fixed: {len(fixed_files)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

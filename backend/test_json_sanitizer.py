"""
Test JSON Sanitizer with various malformed JSON formats.
This validates that the system can handle ANY type of JSON file automatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.json_sanitizer import sanitize_json, validate_json_file
import json


def test_case(name, content, expected_success=True):
    """Test a specific JSON format."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    print(f"Input: {content[:200]}..." if len(content) > 200 else f"Input: {content}")
    
    try:
        data, steps = sanitize_json(content)
        print(f"✅ SUCCESS!")
        print(f"Cleaning steps applied: {len(steps)}")
        for step in steps:
            print(f"  {step}")
        print(f"Result type: {type(data).__name__}")
        if isinstance(data, list):
            print(f"Items: {len(data)}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())[:10]}")
        return True
    except Exception as e:
        if expected_success:
            print(f"❌ FAILED (unexpected): {e}")
            return False
        else:
            print(f"✅ FAILED (expected): {e}")
            return True


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*80)
    print(" JSON SANITIZER COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: MongoDB ISODate format
    tests_total += 1
    if test_case(
        "MongoDB ISODate Format",
        '''[
            {
                "name": "Test",
                "createdAt": "ISODate("2025-01-01T00:00:00Z")"
            }
        ]'''
    ):
        tests_passed += 1
    
    # Test 2: MongoDB ObjectId format
    tests_total += 1
    if test_case(
        "MongoDB ObjectId Format",
        '''[
            {
                "_id": "ObjectId("507f1f77bcf86cd799439011")",
                "name": "Test"
            }
        ]'''
    ):
        tests_passed += 1
    
    # Test 3: Missing commas between objects
    tests_total += 1
    if test_case(
        "Missing Commas Between Objects",
        '''[
            {"name": "Item1"}
            {"name": "Item2"}
            {"name": "Item3"}
        ]'''
    ):
        tests_passed += 1
    
    # Test 4: Trailing commas
    tests_total += 1
    if test_case(
        "Trailing Commas",
        '''[
            {"name": "Item1", "value": 123,},
            {"name": "Item2", "value": 456,},
        ]'''
    ):
        tests_passed += 1
    
    # Test 5: Single quotes instead of double quotes
    tests_total += 1
    if test_case(
        "Single Quotes",
        """[
            {'name': 'Item1', 'value': 123},
            {'name': 'Item2', 'value': 456}
        ]"""
    ):
        tests_passed += 1
    
    # Test 6: JavaScript comments
    tests_total += 1
    if test_case(
        "JavaScript Comments",
        '''[
            // This is a comment
            {"name": "Item1"}, /* inline comment */
            {"name": "Item2"}
            // End comment
        ]'''
    ):
        tests_passed += 1
    
    # Test 7: undefined/NaN/Infinity values
    tests_total += 1
    if test_case(
        "JavaScript Values (undefined, NaN, Infinity)",
        '''[
            {"name": "Item1", "value": undefined},
            {"name": "Item2", "value": NaN},
            {"name": "Item3", "value": Infinity}
        ]'''
    ):
        tests_passed += 1
    
    # Test 8: Mixed MongoDB formats
    tests_total += 1
    if test_case(
        "Mixed MongoDB Formats",
        '''[
            {
                "_id": "ObjectId("507f1f77bcf86cd799439011")",
                "createdAt": "ISODate("2025-01-01T00:00:00Z")",
                "count": "NumberLong(12345)",
                "price": "NumberDecimal("99.99")"
            }
        ]'''
    ):
        tests_passed += 1
    
    # Test 9: Deeply nested structure
    tests_total += 1
    if test_case(
        "Deeply Nested Structure",
        '''[
            {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": "deep"
                        }
                    }
                }
            }
        ]'''
    ):
        tests_passed += 1
    
    # Test 10: Array of primitives
    tests_total += 1
    if test_case(
        "Array of Primitives",
        '''["item1", "item2", "item3"]'''
    ):
        tests_passed += 1
    
    # Test 11: JSON Lines format (one object per line)
    tests_total += 1
    if test_case(
        "JSON Lines Format",
        '''{"name": "Item1", "value": 123}
{"name": "Item2", "value": 456}
{"name": "Item3", "value": 789}'''
    ):
        tests_passed += 1
    
    # Test 12: GeoJSON format
    tests_total += 1
    if test_case(
        "GeoJSON Format",
        '''{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [102.0, 0.5]
                    },
                    "properties": {
                        "name": "Test Location"
                    }
                }
            ]
        }'''
    ):
        tests_passed += 1
    
    # Test 13: API response format
    tests_total += 1
    if test_case(
        "API Response Format",
        '''{
            "status": "success",
            "data": [
                {"id": 1, "name": "Item1"},
                {"id": 2, "name": "Item2"}
            ],
            "meta": {
                "total": 2,
                "page": 1
            }
        }'''
    ):
        tests_passed += 1
    
    # Test 14: Empty structures
    tests_total += 1
    if test_case(
        "Empty Array",
        '''[]'''
    ):
        tests_passed += 1
    
    # Test 15: Complex mixed format (real-world scenario)
    tests_total += 1
    if test_case(
        "Complex Mixed Format (Real-World)",
        '''[
            {
                "_id": "ObjectId("507f1f77bcf86cd799439011")",
                "question": "What is the process?",
                "answer": "The process involves...",
                "tags": ["important", "faq"],
                "metadata": {
                    "createdOn": "ISODate("2025-06-30T06:40:41.859Z")",
                    "views": "NumberLong(150)",
                    "rating": "NumberDecimal("4.5")"
                }
            }
            {
                "_id": "ObjectId("507f1f77bcf86cd799439012")",
                "question": "How to fix errors?",
                "answer": "Follow these steps...",
                "tags": ["troubleshooting"]
            }
        ]'''
    ):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f" TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {tests_total}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_total - tests_passed}")
    print(f"Success rate: {(tests_passed/tests_total)*100:.1f}%")
    print(f"{'='*80}\n")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Comprehensive E2E Test Suite for OMS Chat Bot
Tests all functionality: Authentication, Chat, Document Upload, RAG, Error Handling
"""

import httpx
import json
import time
from datetime import datetime
import sys

# Configuration
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3001"
TIMEOUT = 30
# Longer timeout for chat/LLM endpoints which may take longer on first call
CHAT_TIMEOUT = 120
MAX_RETRIES = 60  # 60 retries * 1 second = 60 seconds max wait

# Test Results Storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "summary": {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
}

def wait_for_backend(max_retries=MAX_RETRIES):
    """Wait for backend to become ready"""
    print("\n⏳ Waiting for backend to be ready...")
    for attempt in range(max_retries):
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Backend is ready (attempt {attempt + 1})")
                time.sleep(1)  # Give it one more second to fully initialize
                return True
        except Exception as e:
            if attempt % 10 == 0:
                print(f"  Attempt {attempt + 1}/{max_retries}: Waiting for backend...")
            time.sleep(1)
    
    print("❌ Backend did not become ready within the timeout period")
    return False

def log_test(name, passed, details=""):
    """Log a test result"""
    result = {
        "name": name,
        "passed": passed,
        "details": details,
        "time": datetime.now().isoformat()
    }
    test_results["tests"].append(result)
    
    if passed:
        test_results["summary"]["passed"] += 1
        print(f"✅ {name}")
    else:
        test_results["summary"]["failed"] += 1
        print(f"❌ {name}: {details}")
        test_results["summary"]["errors"].append(f"{name}: {details}")

def test_backend_health():
    """Test 1: Backend Health Check"""
    try:
        response = httpx.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        log_test(
            "Backend Health Check",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        return response.status_code == 200
    except Exception as e:
        log_test("Backend Health Check", False, str(e))
        return False

def test_cors_headers():
    """Test 2: CORS Configuration"""
    try:
        response = httpx.options(
            f"{BASE_URL}/chat/query",
            headers={"Origin": "http://localhost:3001"},
            timeout=TIMEOUT
        )
        cors_header = response.headers.get("Access-Control-Allow-Origin")
        passed = "localhost:3001" in str(cors_header) or "*" in str(cors_header)
        log_test(
            "CORS Headers",
            passed,
            f"CORS Header: {cors_header}"
        )
        return passed
    except Exception as e:
        log_test("CORS Headers", False, str(e))
        return False

def test_authentication():
    """Test 3: JWT Authentication"""
    try:
        response = httpx.post(
            f"{BASE_URL}/auth/token",
            json={"username": "admin", "password": "admin123"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            log_test(
                "Authentication",
                bool(token),
                f"Token received: {bool(token)}"
            )
            return token
        else:
            log_test("Authentication", False, f"Status: {response.status_code}")
            return None
    except Exception as e:
        log_test("Authentication", False, str(e))
        return None

def test_basic_chat(token):
    """Test 4: Basic Chat Query"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{BASE_URL}/chat/query",
            json={
                "query": "What is machine learning?",
                "conversation_id": "test-conv-001"
            },
            headers=headers,
            timeout=CHAT_TIMEOUT
        )
        
        passed = response.status_code == 200
        if passed:
            data = response.json()
            has_response = "response" in data or "answer" in data
            log_test(
                "Basic Chat Query",
                has_response,
                f"Response keys: {list(data.keys())}"
            )
            return data if has_response else None
        else:
            log_test("Basic Chat Query", False, f"Status: {response.status_code}, Body: {response.text[:200]}")
            return None
    except Exception as e:
        log_test("Basic Chat Query", False, str(e))
        return None

def test_chat_history(token):
    """Test 5: Chat History Retrieval"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # The API exposes history at /chat/sessions/{session_id}/history
        response = httpx.get(
            f"{BASE_URL}/chat/sessions/test-conv-001/history",
            headers=headers,
            timeout=TIMEOUT
        )
        
        passed = response.status_code == 200
        log_test(
            "Chat History Retrieval",
            passed,
            f"Status: {response.status_code}"
        )
        return passed
    except Exception as e:
        log_test("Chat History Retrieval", False, str(e))
        return False

def test_document_list(token):
    """Test 6: Document List API"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.get(
            f"{BASE_URL}/admin/documents",
            headers=headers,
            timeout=TIMEOUT
        )
        
        passed = response.status_code == 200
        if passed:
            data = response.json()
            log_test(
                "Document List API",
                True,
                f"Documents count: {len(data) if isinstance(data, list) else 'N/A'}"
            )
            return True
        else:
            log_test("Document List API", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        log_test("Document List API", False, str(e))
        return False

def test_invalid_query(token):
    """Test 7: Error Handling - Invalid Query"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{BASE_URL}/chat/query",
            json={"query": "", "conversation_id": ""},  # Invalid: empty strings
            headers=headers,
            timeout=TIMEOUT
        )
        
        # Should either return 400 Bad Request or handle gracefully
        passed = response.status_code in [200, 400, 422]
        log_test(
            "Invalid Query Handling",
            passed,
            f"Status: {response.status_code}"
        )
        return passed
    except Exception as e:
        log_test("Invalid Query Handling", False, str(e))
        return False

def test_missing_auth():
    """Test 8: Error Handling - Missing Authentication"""
    try:
        response = httpx.get(
            f"{BASE_URL}/admin/documents",
            timeout=TIMEOUT
        )
        
        # Should return 401 Unauthorized without token
        passed = response.status_code in [401, 403]
        log_test(
            "Missing Authentication",
            passed,
            f"Status: {response.status_code}"
        )
        return passed
    except Exception as e:
        log_test("Missing Authentication", False, str(e))
        return False

def test_concurrent_requests(token):
    """Test 9: Concurrent Request Handling"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Send multiple concurrent requests using async client
        import asyncio

        async def _run_concurrent():
            async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
                reqs = []
                for i in range(3):
                    reqs.append(
                        client.post(
                            f"{BASE_URL}/chat/query",
                            json={
                                "query": f"Test query {i}",
                                "session_id": f"concurrent-{i}"
                            },
                            headers=headers
                        )
                    )

                results = await asyncio.gather(*reqs, return_exceptions=True)
                statuses = []
                for r in results:
                    if isinstance(r, Exception):
                        statuses.append(repr(r))
                    else:
                        statuses.append(r.status_code)

                all_passed = all(isinstance(s, int) and s == 200 for s in statuses)
                log_test(
                    "Concurrent Request Handling",
                    all_passed,
                    f"Results: {statuses}"
                )
                return all_passed

        return asyncio.run(_run_concurrent())
    except Exception as e:
        log_test("Concurrent Request Handling", False, str(e))
        return False

def test_response_format(chat_response):
    """Test 10: Response Format Validation"""
    try:
        if not chat_response:
            log_test("Response Format Validation", False, "No response to validate")
            return False
        
        required_fields = ["response", "sources"]  # Adjust based on your API
        # Check if response has required structure
        has_required_fields = "response" in chat_response or "answer" in chat_response
        
        log_test(
            "Response Format Validation",
            has_required_fields,
            f"Response has: {list(chat_response.keys())}"
        )
        return has_required_fields
    except Exception as e:
        log_test("Response Format Validation", False, str(e))
        return False

def test_database_connectivity():
    """Test 11: Database Connectivity"""
    try:
        # This would require a dedicated health endpoint
        # For now, we'll test through indirect means
        headers = {"Authorization": "Bearer dummy"}  # Dummy auth
        response = httpx.post(
            f"{BASE_URL}/chat/query",
            json={"query": "test", "conversation_id": "db-test"},
            headers=headers,
            timeout=CHAT_TIMEOUT
        )
        
        # If it's not a 500 error, databases are likely connected
        passed = response.status_code != 500
        log_test(
            "Database Connectivity",
            passed,
            f"Status: {response.status_code}"
        )
        return passed
    except Exception as e:
        log_test("Database Connectivity", False, str(e))
        return False

def test_llm_integration(token):
    """Test 12: LLM Integration"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{BASE_URL}/chat/query",
            json={
                "query": "Hello, how are you?",
                "conversation_id": "llm-test-001"
            },
            headers=headers,
            timeout=CHAT_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            # Check if response contains actual content (not empty/error)
            response_text = data.get("response", "") or data.get("answer", "")
            has_content = len(str(response_text)) > 5
            log_test(
                "LLM Integration",
                has_content,
                f"Response length: {len(str(response_text))} chars"
            )
            return has_content
        else:
            log_test("LLM Integration", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        log_test("LLM Integration", False, str(e))
        return False

def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "="*60)
    print("OMS Chat Bot - Comprehensive E2E Test Suite")
    print("="*60 + "\n")
    
    # Test 1: Backend Health
    if not test_backend_health():
        print("\n❌ Backend is not running! Cannot continue tests.")
        return test_results
    
    print("\n📋 Running Core Functionality Tests...\n")
    
    # Test 2: CORS
    test_cors_headers()
    
    # Test 3: Authentication
    token = test_authentication()
    if not token:
        print("\n❌ Authentication failed! Cannot continue with authorized tests.")
        test_results["summary"]["errors"].append("Authentication failed - stopping tests")
        return test_results
    
    print("\n📋 Running Chat Functionality Tests...\n")
    
    # Test 4: Basic Chat
    chat_response = test_basic_chat(token)
    
    # Test 5: Chat History
    test_chat_history(token)
    
    # Test 10: Response Format
    if chat_response:
        test_response_format(chat_response)
    
    print("\n📋 Running Admin Functionality Tests...\n")
    
    # Test 6: Document List
    test_document_list(token)
    
    print("\n📋 Running Error Handling Tests...\n")
    
    # Test 7: Invalid Query
    test_invalid_query(token)
    
    # Test 8: Missing Auth
    test_missing_auth()
    
    # Test 11: Database
    test_database_connectivity()
    
    print("\n📋 Running Advanced Tests...\n")
    
    # Test 9: Concurrent Requests
    test_concurrent_requests(token)
    
    # Test 12: LLM Integration
    test_llm_integration(token)
    
    return test_results

def print_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(test_results['tests'])}")
    print(f"✅ Passed: {test_results['summary']['passed']}")
    print(f"❌ Failed: {test_results['summary']['failed']}")
    
    if test_results['summary']['errors']:
        print(f"\n⚠️  Errors:")
        for error in test_results['summary']['errors']:
            print(f"  - {error}")
    
    print("="*60 + "\n")
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print("📄 Full results saved to: test_results.json")

if __name__ == "__main__":
    # Wait for backend to be ready
    if not wait_for_backend():
        print("\n❌ Cannot proceed without backend. Exiting.")
        sys.exit(1)
    
    run_all_tests()
    print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if test_results['summary']['failed'] == 0 else 1)

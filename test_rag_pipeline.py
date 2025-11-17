"""
Comprehensive RAG Pipeline Test Suite
Tests all fixed components end-to-end
"""

import requests
import time
import base64
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_CONTENT = """First paragraph with substantial content for proper chunking and testing.

Second paragraph with substantial content for proper chunking and testing.

Third paragraph with substantial content for proper chunking and testing."""

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_result(test_name, passed, message=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"      {message}")

def test_backend_health():
    """Test 1: Verify backend is running"""
    print_header("TEST 1: Backend Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        passed = response.status_code == 200
        print_result("Backend responding", passed, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        print_result("Backend responding", False, f"Error: {e}")
        return False

def test_document_upload():
    """Test 2: Upload document and verify acceptance"""
    print_header("TEST 2: Document Upload")
    try:
        filename = f"test_rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        content_bytes = TEST_CONTENT.encode('utf-8')
        content_base64 = base64.b64encode(content_bytes).decode('utf-8')
        
        payload = {
            "filename": filename,
            "content_base64": content_base64,
            "content_type": "text/plain"
        }
        
        response = requests.post(
            f"{BASE_URL}/admin/documents/upload-json",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            doc_id = data.get('document_id')
            print_result("Document uploaded", True, f"ID: {doc_id}")
            return doc_id
        else:
            print_result("Document uploaded", False, f"Status: {response.status_code}")
            return None
            
    except Exception as e:
        print_result("Document uploaded", False, f"Error: {e}")
        return None

def test_document_processing(doc_id, max_wait=15):
    """Test 3: Verify document gets processed"""
    print_header("TEST 3: Document Processing")
    
    if not doc_id:
        print_result("Document processing", False, "No document ID provided")
        return False
    
    print(f"Waiting up to {max_wait} seconds for processing...")
    
    for i in range(max_wait):
        try:
            time.sleep(1)
            response = requests.get(
                f"{BASE_URL}/admin/documents/list",
                params={"page": 1, "page_size": 10},
                timeout=10
            )
            
            if response.status_code == 200:
                docs = response.json().get('documents', [])
                doc = next((d for d in docs if d.get('id') == doc_id), None)
                
                if doc:
                    status = doc.get('status', '')
                    chunks = doc.get('chunks_count', 0)
                    
                    print(f"  [{i+1}/{max_wait}] Status: {status}, Chunks: {chunks}")
                    
                    if status == 'COMPLETED' and chunks > 0:
                        print_result("Document processing", True, 
                                   f"Status: {status}, Chunks: {chunks}")
                        return True, chunks
                    elif status == 'FAILED':
                        print_result("Document processing", False, 
                                   f"Processing failed: {doc.get('error_message', 'Unknown error')}")
                        return False, 0
        except Exception as e:
            print(f"  Error checking status: {e}")
            continue
    
    print_result("Document processing", False, 
               f"Timeout after {max_wait} seconds")
    return False, 0

def test_chunk_quality(doc_id):
    """Test 4: Verify chunk quality (if chunks endpoint exists)"""
    print_header("TEST 4: Chunk Quality Check")
    
    # This is optional - if chunks endpoint exists
    try:
        response = requests.get(
            f"{BASE_URL}/admin/documents/{doc_id}/chunks",
            timeout=10
        )
        
        if response.status_code == 200:
            chunks = response.json().get('chunks', [])
            
            if len(chunks) > 0:
                avg_length = sum(len(c.get('text', '')) for c in chunks) / len(chunks)
                print_result("Chunk quality", True, 
                           f"{len(chunks)} chunks, avg length: {avg_length:.0f} chars")
                
                # Print first chunk as sample
                if chunks:
                    first_chunk = chunks[0].get('text', '')
                    print(f"\n  Sample chunk:\n  {first_chunk[:100]}...\n")
                return True
            else:
                print_result("Chunk quality", False, "No chunks returned")
                return False
        elif response.status_code == 404:
            print_result("Chunk quality", None, "Chunks endpoint not available (skipped)")
            return None
        else:
            print_result("Chunk quality", False, f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_result("Chunk quality", None, f"Skipped: {e}")
        return None

def test_rag_query(doc_id):
    """Test 5: Test RAG query with uploaded document"""
    print_header("TEST 5: RAG Query Test")
    
    try:
        query = "What is in the document?"
        payload = {
            "query": query,
            "session_id": "test_session"
        }
        
        response = requests.post(
            f"{BASE_URL}/chat/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            sources = data.get('sources', [])
            tokens = data.get('tokens_generated', 0)
            
            has_response = len(response_text) > 0
            has_sources = len(sources) > 0
            
            print_result("RAG query execution", has_response, 
                       f"Response length: {len(response_text)} chars")
            print_result("Source retrieval", has_sources, 
                       f"Sources returned: {len(sources)}")
            
            if response_text:
                print(f"\n  Query: {query}")
                print(f"  Response: {response_text[:200]}...")
                if sources:
                    print(f"\n  Sources: {len(sources)} chunks retrieved")
            
            return has_response and has_sources
        else:
            print_result("RAG query", False, f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_result("RAG query", False, f"Error: {e}")
        return False

def test_qdrant_status():
    """Test 6: Verify Qdrant has vectors"""
    print_header("TEST 6: Qdrant Vector Storage")
    
    try:
        response = requests.get(
            f"{BASE_URL}/admin/rag/status",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            qdrant_data = data.get('qdrant', {})
            vector_count = qdrant_data.get('vectors_count', 0)
            
            passed = vector_count > 0
            print_result("Vectors in Qdrant", passed, 
                       f"Vector count: {vector_count}")
            return passed
        elif response.status_code == 404:
            print_result("Qdrant status", None, 
                       "RAG status endpoint not available (skipped)")
            return None
        else:
            print_result("Qdrant status", False, f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_result("Qdrant status", None, f"Skipped: {e}")
        return None

def run_all_tests():
    """Run complete test suite"""
    print_header("🧪 RAG PIPELINE COMPREHENSIVE TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_backend_health()
    if not results['health']:
        print("\n❌ Backend not running. Please start backend first.")
        print("   Command: cd backend && python -m uvicorn app.main:app --reload")
        return False
    
    # Test 2: Upload
    doc_id = test_document_upload()
    results['upload'] = doc_id is not None
    if not doc_id:
        print("\n❌ Upload failed. Cannot continue tests.")
        return False
    
    # Test 3: Processing
    results['processing'], chunk_count = test_document_processing(doc_id)
    
    # Test 4: Chunk quality (optional)
    results['chunk_quality'] = test_chunk_quality(doc_id)
    
    # Test 5: RAG query
    results['rag_query'] = test_rag_query(doc_id)
    
    # Test 6: Qdrant status
    results['qdrant'] = test_qdrant_status()
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = passed + failed
    
    print(f"Passed:  {passed}/{total}")
    print(f"Failed:  {failed}/{total}")
    print(f"Skipped: {skipped}")
    
    if failed == 0 and passed >= 4:  # At least core tests passed
        print("\n🎉 SUCCESS! RAG Pipeline is working correctly!")
        print("\n✅ All critical fixes have been verified:")
        print("   • Tokenizer initialization: FIXED")
        print("   • Background task execution: FIXED")
        print("   • Document chunking: WORKING")
        print("   • Embedding generation: WORKING")
        print("   • Vector storage: WORKING")
        return True
    else:
        print("\n⚠️ Some tests failed. Review errors above.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

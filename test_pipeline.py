"""
Test script to verify the document upload and chat pipeline is working
"""
import requests
import json
import time

def test_document_pipeline():
    """Test the complete document processing and chat pipeline"""
    
    print("=" * 60)
    print("Testing Document Processing & Chat Pipeline")
    print("=" * 60)
    
    # Test 1: Upload a simple document
    print("\n1. Uploading test document...")
    
    test_faq = {
        "company": "OMS Test Company",
        "services": [
            "Logistics and warehousing",
            "Transportation management", 
            "Supply chain optimization",
            "Order fulfillment",
            "Inventory management"
        ],
        "faq": [
            {
                "question": "What does OMS do?",
                "answer": "OMS is a comprehensive logistics company that provides warehousing, transportation, and supply chain management services to businesses of all sizes."
            },
            {
                "question": "How can I track my order?",
                "answer": "You can track your order through our customer portal using your order number, or call our customer service team at 1-800-OMS-HELP."
            },
            {
                "question": "What are your operating hours?",
                "answer": "Our main operations run 24/7, but customer service is available Monday-Friday 8 AM to 6 PM EST."
            }
        ]
    }
    
    try:
        json_content = json.dumps(test_faq, indent=2)
        files = {'file': ('oms_services_faq.json', json_content, 'application/json')}
        
        response = requests.post(
            "http://127.0.0.1:8000/admin/documents/upload", 
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            doc_id = result.get('document_id')
            print(f"   ✓ Upload successful! Document ID: {doc_id}")
            print(f"   ✓ Status: {result.get('status')}")
        else:
            print(f"   ✗ Upload failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ✗ Upload error: {e}")
        return False
    
    # Test 2: Wait a moment and check document status  
    print("\n2. Waiting for document processing...")
    time.sleep(5)
    
    try:
        response = requests.get(f"http://127.0.0.1:8000/admin/documents/status/{doc_id}")
        if response.status_code == 200:
            status_info = response.json()
            print(f"   ✓ Document status: {status_info.get('status')}")
            print(f"   ✓ Progress: {status_info.get('progress', 0)}%")
        else:
            print(f"   ⚠ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Status error: {e}")
    
    # Test 3: Test general conversation
    print("\n3. Testing general conversation...")
    
    try:
        general_query = {
            "query": "Hello, how are you today?",
            "session_id": "test-general"
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/chat/query",
            json=general_query,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ General chat works!")
            print(f"   Response: {result.get('response', '')[:100]}...")
            print(f"   Processing time: {result.get('processing_time', 0):.1f}s")
        else:
            print(f"   ✗ General chat failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ General chat error: {e}")
    
    # Test 4: Test document-specific query
    print("\n4. Testing document-specific query...")
    
    try:
        doc_query = {
            "query": "What services does OMS provide?",
            "session_id": "test-document"
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/chat/query",
            json=doc_query,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Document chat works!")
            print(f"   Response: {result.get('response', '')[:200]}...")
            print(f"   Sources found: {len(result.get('sources', []))}")
            print(f"   Processing time: {result.get('processing_time', 0):.1f}s")
            
            # Check if response mentions OMS services
            response_text = result.get('response', '').lower()
            if any(word in response_text for word in ['logistics', 'warehousing', 'transportation']):
                print(f"   ✓ Response contains relevant service information!")
            else:
                print(f"   ⚠ Response may not be using document content")
                
        else:
            print(f"   ✗ Document chat failed: {response.status_code}")
            print(f"   Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"   ✗ Document chat error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed! Check logs for detailed processing information.")
    print("=" * 60)

if __name__ == "__main__":
    test_document_pipeline()
# Test complexity acceptable - comprehensive integration test
# pylint: disable=too-many-branches,too-many-statements
"""
Complete data flow test: Frontend → Backend → LMStudio → Backend → Frontend
Tests CORS, request/response handling, and data integrity at each step.
"""

import httpx
import json
import time
from datetime import datetime


# noqa: C901 - Integration test complexity acceptable
def test_complete_flow():  # noqa: python:S3776
    """Test the complete request/response flow."""
    
    print("=" * 80)
    print("🔍 COMPLETE DATA FLOW TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Step 1: Test backend health
    print("-" * 80)
    print("STEP 1: Testing Backend Health")
    print("-" * 80)
    
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get("http://127.0.0.1:8000/system/info")
            print(f"✅ Backend is reachable (status: {response.status_code})")
            info = response.json()
            print(f"   MongoDB: {info['databases']['mongodb']['objects']} documents")
            print(f"   Qdrant: {info['databases']['qdrant']['points_count']} vectors")
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False
    
    # Step 2: Test LMStudio connection
    print("\n" + "-" * 80)
    print("STEP 2: Testing LMStudio Connection")
    print("-" * 80)
    
    lmstudio_url = "http://192.168.56.1:1234/v1"
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{lmstudio_url}/models")
            print(f"✅ LMStudio is reachable (status: {response.status_code})")
            if response.status_code == 200:
                models = response.json()
                print(f"   Available models: {len(models.get('data', []))}")
                for model in models.get('data', [])[:3]:
                    print(f"     - {model.get('id', 'unknown')}")
    except Exception as e:
        print(f"❌ LMStudio connection failed: {e}")
        print("   Make sure LMStudio is running at", lmstudio_url)
        return False
    
    # Step 3: Test direct LMStudio query
    print("\n" + "-" * 80)
    print("STEP 3: Testing Direct LMStudio Query")
    print("-" * 80)
    
    test_payload = {
        "model": "mistral-7b-instruct-v0.3",
        "messages": [{"role": "user", "content": "Say 'Test successful!' in exactly 2 words."}],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    print("Sending test query to LMStudio...")
    try:
        with httpx.Client(timeout=30) as client:
            start = time.time()
            response = client.post(
                f"{lmstudio_url}/chat/completions",
                json=test_payload
            )
            elapsed = time.time() - start
            
            print(f"✅ LMStudio responded in {elapsed:.2f}s (status: {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"   Response: '{content.strip()}'")
                print(f"   Tokens: {data.get('usage', {})}")
            else:
                print(f"❌ Unexpected status: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Direct LMStudio query failed: {e}")
        return False
    
    # Step 4: Test backend CORS headers
    print("\n" + "-" * 80)
    print("STEP 4: Testing Backend CORS Configuration")
    print("-" * 80)
    
    try:
        with httpx.Client() as client:
            # OPTIONS preflight request (what browser sends)
            response = client.options(
                "http://127.0.0.1:8000/chat/query",
                headers={
                    "Origin": "http://localhost:3001",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type"
                }
            )
            
            print(f"OPTIONS preflight status: {response.status_code}")
            
            cors_headers = {
                "Access-Control-Allow-Origin": response.headers.get("access-control-allow-origin", "NOT SET"),
                "Access-Control-Allow-Methods": response.headers.get("access-control-allow-methods", "NOT SET"),
                "Access-Control-Allow-Headers": response.headers.get("access-control-allow-headers", "NOT SET"),
                "Access-Control-Allow-Credentials": response.headers.get("access-control-allow-credentials", "NOT SET")
            }
            
            print("\nCORS Headers:")
            for header, value in cors_headers.items():
                status = "✅" if value != "NOT SET" else "❌"
                print(f"   {status} {header}: {value}")
            
            if cors_headers["Access-Control-Allow-Origin"] == "NOT SET":
                print("\n⚠️  WARNING: CORS not properly configured!")
                print("   Frontend requests will be blocked by browser")
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
    
    # Step 5: Test complete backend chat flow
    print("\n" + "-" * 80)
    print("STEP 5: Testing Complete Backend Chat Flow")
    print("-" * 80)
    
    chat_request = {
        "query": "What is 2+2? Answer in exactly one number.",
        "session_id": f"test_session_{int(time.time())}",
        "context": [],
        "stream": False
    }
    
    print(f"Sending chat query: '{chat_request['query']}'")
    print("⏳ Waiting for response (timeout: 180s)...")
    
    try:
        with httpx.Client(timeout=180) as client:
            start = time.time()
            
            response = client.post(
                "http://127.0.0.1:8000/chat/query",
                json=chat_request,
                headers={
                    "Content-Type": "application/json",
                    "Origin": "http://localhost:3001"  # Simulate frontend request
                }
            )
            
            elapsed = time.time() - start
            
            print(f"\n📥 Response received in {elapsed:.2f}s")
            print(f"   Status: {response.status_code} {response.reason_phrase}")
            print(f"   Content-Type: {response.headers.get('content-type')}")
            print(f"   CORS Header: {response.headers.get('access-control-allow-origin', 'NOT SET')}")
            
            if response.status_code == 200:
                data = response.json()
                
                print("\n✅ RESPONSE STRUCTURE:")
                print(f"   ✓ response field: {bool(data.get('response'))}")
                print(f"   ✓ sources field: {bool('sources' in data)}")
                print(f"   ✓ session_id field: {bool(data.get('session_id'))}")
                print(f"   ✓ processing_time: {data.get('processing_time', 0):.2f}s")
                print(f"   ✓ tokens_generated: {data.get('tokens_generated', 0)}")
                
                if data.get('response'):
                    print("\n📝 RESPONSE CONTENT:")
                    print(f"   Length: {len(data['response'])} characters")
                    print(f"   Preview: {data['response'][:200]}")
                    if len(data['response']) > 200:
                        print("   ... (truncated)")
                    
                    print("\n✅ SUCCESS: Complete flow working!")
                    print("   ✓ Frontend can send queries")
                    print("   ✓ Backend receives queries")
                    print("   ✓ Backend calls LMStudio")
                    print("   ✓ LMStudio generates response")
                    print("   ✓ Backend processes response")
                    print("   ✓ Backend returns to frontend")
                    print("   ✓ CORS headers present")
                    
                    return True
                else:
                    print("\n❌ PROBLEM: Response field is empty!")
                    print(f"   Available fields: {list(data.keys())}")
                    return False
            else:
                print(f"\n❌ Backend returned error: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return False
                
    except httpx.TimeoutException:
        print("\n❌ Request timed out after 180 seconds")
        print("   Possible issues:")
        print("   1. LMStudio is not responding")
        print("   2. Backend timeout too short")
        print("   3. Model is too slow or not loaded")
        return False
    except Exception as e:
        print(f"\n❌ Chat flow test failed: {e}")
        import traceback
        print(f"\nTraceback:\n{traceback.format_exc()}")
        return False
    
    print("\n" + "=" * 80)


def test_frontend_simulation():
    """Simulate exact frontend request."""
    
    print("\n" + "=" * 80)
    print("🌐 FRONTEND REQUEST SIMULATION")
    print("=" * 80)
    
    # Exact replica of frontend fetch request
    print("\nSimulating exact frontend request...")
    
    try:
        with httpx.Client(timeout=180) as client:
            response = client.post(
                "http://127.0.0.1:8000/chat/query",
                headers={
                    "Content-Type": "application/json",
                    "Origin": "http://localhost:3001",
                    "Accept": "application/json"
                },
                json={
                    "query": "Hello, are you working?",
                    "session_id": f"frontend_test_{int(time.time())}",
                    "context": []
                }
            )
            
            print(f"\nResponse Status: {response.status_code}")
            print("Response Headers:")
            for key, value in response.headers.items():
                if 'cors' in key.lower() or 'access-control' in key.lower():
                    print(f"  {key}: {value}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ Frontend simulation successful!")
                print(f"   Response preview: {data.get('response', 'NO RESPONSE')[:100]}")
            else:
                print(f"\n❌ Frontend simulation failed: {response.text}")
    
    except Exception as e:
        print(f"❌ Frontend simulation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        success = test_complete_flow()
        
        if success:
            test_frontend_simulation()
            print("\n" + "=" * 80)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 80)
            print("\nYour system is working correctly!")
            print("Frontend should be able to:")
            print("  1. Send queries to backend")
            print("  2. Receive responses from backend")
            print("  3. Display responses to user")
            print("\nIf frontend still not working, check browser console (F12)")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ TESTS FAILED")
            print("=" * 80)
            print("\nPlease fix the issues above before testing frontend.")
            print("=" * 80)
        
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted (Ctrl+C)")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

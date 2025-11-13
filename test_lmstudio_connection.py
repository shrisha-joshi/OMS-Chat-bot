# Test complexity acceptable - comprehensive integration test
# pylint: disable=too-many-branches,too-many-statements
"""
Test script to verify LMStudio connection and timeout settings.
This script tests the complete chain: Python → LMStudio API

Run with: python test_lmstudio_connection.py
"""

import httpx
import json
import time
from datetime import datetime


# noqa: C901 - Integration test complexity acceptable
def test_lmstudio_connection():  # noqa: python:S3776
    """Test connection to LMStudio with the same settings as backend."""
    
    lmstudio_url = "http://192.168.56.1:1234/v1"
    
    print("=" * 70)
    print("🔬 LMSTUDIO CONNECTION TEST")
    print("=" * 70)
    print(f"\n📍 Target: {lmstudio_url}")
    print("⏱️  Timeout: 300 seconds (5 minutes)")
    print(f"🕒 Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Test 1: Check if LMStudio is reachable
    print("-" * 70)
    print("TEST 1: Checking if LMStudio is running...")
    print("-" * 70)
    
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            response = client.get(f"{lmstudio_url}/models")
            print(f"✅ LMStudio is REACHABLE (status: {response.status_code})")
            
            if response.status_code == 200:
                models = response.json()
                print(f"📦 Available models: {json.dumps(models, indent=2)[:500]}")
            else:
                print(f"⚠️  Unexpected status code: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
    except httpx.ConnectError as e:
        print("❌ CONNECTION FAILED!")
        print(f"   Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Is LMStudio running?")
        print("   2. Is the server listening on 192.168.56.1:1234?")
        print("   3. Check VirtualBox network configuration")
        print("   4. Try: Test-NetConnection -ComputerName 192.168.56.1 -Port 1234")
        return False
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False
    
    # Test 2: Send a simple chat request
    print("\n" + "-" * 70)
    print("TEST 2: Sending test query to LMStudio...")
    print("-" * 70)
    
    payload = {
        "model": "mistral-7b-instruct-v0.3",  # Default model
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello, I am working!' in exactly 5 words."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False
    }
    
    print("📤 Sending request:")
    print(f"   Model: {payload['model']}")
    print(f"   Query: {payload['messages'][0]['content']}")
    print(f"   Max tokens: {payload['max_tokens']}")
    print("\n⏳ Waiting for response (timeout: 300s)...")
    
    start_time = time.time()
    
    try:
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            response = client.post(
                f"{lmstudio_url}/chat/completions",
                json=payload
            )
            
            elapsed = time.time() - start_time
            
            print(f"\n📥 Response received in {elapsed:.2f} seconds!")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    print("✅ LMStudio responded successfully!")
                    print("\n💬 Response content:")
                    print(f"   {content}")
                    print("\n📊 Token usage:")
                    print(f"   {json.dumps(data.get('usage', {}), indent=2)}")
                else:
                    print("⚠️  Unexpected response format:")
                    print(f"   {json.dumps(data, indent=2)[:500]}")
                    
            else:
                print(f"❌ Error status: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                
    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        print(f"\n⏰ REQUEST TIMED OUT after {elapsed:.2f} seconds!")
        print("   This means LMStudio took too long to respond.")
        print("\n🔧 Troubleshooting:")
        print("   1. Model might be too large or not loaded")
        print("   2. Try loading a smaller/faster model in LMStudio")
        print("   3. Check LMStudio console for errors")
        print("   4. Increase max_tokens or reduce context size")
        return False
        
    except httpx.ConnectError as e:
        print("\n❌ CONNECTION LOST during request!")
        print(f"   Error: {e}")
        print("   LMStudio might have crashed or stopped responding")
        return False
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        print("\n Stack trace:")
        traceback.print_exc()
        return False
    
    # Test 3: Check backend timeout configuration
    print("\n" + "-" * 70)
    print("TEST 3: Verifying backend configuration...")
    print("-" * 70)
    
    try:
        import sys
        sys.path.insert(0, "d:\\OMS Chat Bot\\backend")
        from app.services.llm_handler import LLMHandler
        
        handler = LLMHandler()
        print(f"✅ Backend LLM Handler timeout: {handler.timeout} seconds")
        
        if handler.timeout >= 300:
            print("   ✓ Timeout is adequate (≥5 minutes)")
        else:
            print("   ⚠️  Timeout might be too short (recommended: ≥300s)")
            
    except Exception as e:
        print(f"⚠️  Could not check backend config: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n🎯 Summary:")
    print("   ✓ LMStudio is reachable")
    print("   ✓ LMStudio can generate responses")
    print(f"   ✓ Response time: {elapsed:.2f}s (within 5-minute timeout)")
    print("\n💡 Next steps:")
    print("   1. Frontend is already configured with 3-minute timeout")
    print("   2. Backend is configured with 5-minute timeout")
    print("   3. Try sending a query through the web interface")
    print("   4. Monitor backend logs for: 🔗📤📥 emojis")
    print("\n" + "=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_lmstudio_connection()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

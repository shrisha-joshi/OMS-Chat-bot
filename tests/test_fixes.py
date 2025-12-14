import asyncio
import httpx
import os
import shutil
import time
import sys

BASE_URL = "http://127.0.0.1:8001"

async def test_chat():
    print("\n--- Testing Chat API ---")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat/query",
                json={"query": "Hello, are you working?", "session_id": "test-session"}
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Response:", response.json().get("response")[:50] + "...")
                return True
            else:
                print("Error:", response.text)
                return False
        except Exception as e:
            print(f"Request failed: {e}")
            return False

async def test_rate_limit():
    print("\n--- Testing Rate Limit ---")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(15):
            try:
                response = await client.post(
                    f"{BASE_URL}/chat/query",
                    json={"query": "spam", "session_id": "spam-session"}
                )
                print(f"Req {i+1}: {response.status_code}")
                if response.status_code == 429:
                    print("✅ Rate limit triggered successfully (429)")
                    return True
            except Exception as e:
                print(f"Req {i+1} failed: {type(e).__name__}: {e}")
    print("❌ Rate limit NOT triggered")
    return False

async def test_chunk_upload():
    print("\n--- Testing Chunk Upload ---")
    file_id = f"test_upload_{int(time.time())}"
    filename = "test_large_file.txt"
    content = b"A" * 1024 * 1024 * 2  # 2MB dummy content
    chunk_size = 1024 * 512  # 512KB chunks
    
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    print(f"Uploading {len(chunks)} chunks for {filename}...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Upload chunks
        for i, chunk in enumerate(chunks):
            files = {'file': (str(i), chunk)}
            data = {'file_id': file_id, 'chunk_index': str(i)}
            resp = await client.post(f"{BASE_URL}/admin/documents/upload-chunk", files=files, data=data)
            if resp.status_code != 200:
                print(f"❌ Chunk {i} failed: {resp.status_code} {resp.text}")
                return False
            print(f"Chunk {i} uploaded: {resp.status_code}")

        # Assemble
        print("Assembling file...")
        data = {'file_id': file_id, 'filename': filename}
        resp = await client.post(f"{BASE_URL}/admin/documents/assemble", data=data)
        
        if resp.status_code == 200:
            print("✅ Assembly success:", resp.json())
            return True
        else:
            print(f"❌ Assembly failed: {resp.status_code} {resp.text}")
            return False

async def main():
    print("Waiting for server to be ready...")
    await asyncio.sleep(5)
    
    chat_ok = await test_chat()
    rate_ok = await test_rate_limit()
    
    print("Sleeping for 60s to reset rate limit...")
    await asyncio.sleep(60)
    
    upload_ok = await test_chunk_upload()
    
    if chat_ok and rate_ok and upload_ok:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

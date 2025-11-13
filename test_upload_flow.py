#!/usr/bin/env python3
# Test complexity acceptable - comprehensive integration test
# pylint: disable=too-many-branches,too-many-statements
"""
Test script to verify the complete document upload and chunking flow.
This script will:
1. Upload a test document
2. Monitor ingestion progress
3. Verify chunks are created
4. Verify embeddings are stored in Qdrant
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_FILE_PATH = Path("test_document.txt")

# Create a test document if it doesn't exist
if not TEST_FILE_PATH.exists():
    test_content = """
    # Test Document for Upload Flow Verification
    
    ## Introduction
    This is a test document to verify the document upload and chunking flow in the RAG system.
    The document contains multiple sections to ensure proper chunking behavior.
    
    ## Section 1: Basic Information
    The system should chunk this content and create embeddings for each chunk.
    Chunks should be stored in MongoDB and embeddings should be indexed in Qdrant.
    
    ## Section 2: Entity Information
    Important people: John Smith, Jane Doe, Bob Wilson
    Important organizations: Acme Corp, TechCorp, FutureTech
    Important locations: New York, San Francisco, London
    
    ## Section 3: Complex Content
    This section contains more text to ensure the chunking algorithm works correctly.
    The chunks should maintain semantic meaning and include proper overlap.
    Multiple chunks should be created from this document.
    
    ## Section 4: Verification Points
    - Documents should be stored in GridFS
    - Chunks should be created in MongoDB chunks collection
    - Embeddings should be generated and stored in Qdrant
    - Entity relationships should be extracted and stored in MongoDB
    - Ingestion logs should track all processing steps
    """
    
    with open(TEST_FILE_PATH, "w") as f:
        f.write(test_content)
    print(f"✓ Created test document: {TEST_FILE_PATH}")

# noqa: C901 - Integration test complexity acceptable
async def test_upload_flow():  # noqa: python:S3776
    """Main test function."""
    async with aiohttp.ClientSession() as session:
        print("\n" + "="*80)
        print("DOCUMENT UPLOAD AND CHUNKING FLOW TEST")
        print("="*80)
        
        # Step 1: Upload document
        print("\n[Step 1] Uploading document...")
        # Note: Sync file ops acceptable in test setup
        # Note: Sync file ops acceptable in test setup
        with open(TEST_FILE_PATH, "rb") as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=TEST_FILE_PATH.name)
            
            try:
                async with session.post(f"{BASE_URL}/admin/documents/upload", data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        doc_id = result.get("document_id")
                        print("✓ Upload successful")
                        print(f"  - Document ID: {doc_id}")
                        print(f"  - Filename: {result.get('filename')}")
                        print(f"  - Size: {result.get('size')} bytes")
                        print(f"  - Status: {result.get('status')}")
                    else:
                        print(f"✗ Upload failed with status {resp.status}")
                        print(f"  Response: {await resp.text()}")
                        return
            except Exception as e:
                print(f"✗ Upload error: {e}")
                return
        
        # Step 2: Wait for processing and check status
        print("\n[Step 2] Monitoring ingestion progress...")
        max_wait = 60  # Wait up to 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                async with session.get(f"{BASE_URL}/admin/documents/status/{doc_id}") as resp:
                    if resp.status == 200:
                        status = await resp.json()
                        current_stage = status.get("current_stage", "UNKNOWN")
                        progress = status.get("overall_progress", 0)
                        chunks_count = status.get("chunks_count", 0)
                        
                        print(f"  Status: {current_stage} ({progress}%)", end="\r")
                        
                        # Check for completion
                        if current_stage in ["COMPLETE", "FAILED"]:
                            print(f"\n✓ Processing completed: {current_stage}")
                            print("\n  Detailed Status:")
                            print(f"  - Overall Progress: {progress}%")
                            print(f"  - Chunks Created: {chunks_count}")
                            print(f"  - Ingestion Status: {status.get('ingest_status')}")
                            
                            # Print stages
                            stages = status.get("stages", [])
                            print("\n  Processing Stages:")
                            for stage in stages:
                                stage_name = stage.get("name")
                                stage_status = stage.get("status")
                                message = stage.get("message")
                                print(f"    - {stage_name}: {stage_status}")
                                if message:
                                    print(f"      Message: {message}")
                            
                            if status.get("error_message"):
                                print(f"\n  Error: {status.get('error_message')}")
                            
                            break
                    else:
                        print(f"✗ Status check failed: {resp.status}")
                        break
            except Exception as e:
                print(f"✗ Status check error: {e}")
                break
            
            await asyncio.sleep(2)
        else:
            print(f"\n✗ Processing timeout after {max_wait} seconds")
        
        # Step 3: Verify chunks were created
        print("\n[Step 3] Verifying chunks in MongoDB...")
        try:
            async with session.get(f"{BASE_URL}/admin/documents/list") as resp:
                if resp.status == 200:
                    doc_list = await resp.json()
                    docs = doc_list.get("documents", [])
                    
                    for doc in docs:
                        if doc.get("id") == doc_id:
                            chunks = doc.get("chunks_count", 0)
                            if chunks > 0:
                                print(f"✓ Chunks found: {chunks}")
                            else:
                                print("✗ No chunks found for document")
                            break
        except Exception as e:
            print(f"✗ Failed to verify chunks: {e}")
        
        # Step 4: Test retrieval
        print("\n[Step 4] Testing retrieval and embedding...")
        try:
            test_query = "What is the test document about?"
            query_data = {"query": test_query}
            
            async with session.post(f"{BASE_URL}/chat/query", json=query_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    sources = result.get("sources", [])
                    if sources:
                        print(f"✓ Retrieval successful: {len(sources)} sources found")
                        for i, source in enumerate(sources[:3], 1):
                            print(f"  - Source {i}: {source.get('filename')} (score: {source.get('score', 0):.3f})")
                    else:
                        print("✗ No sources retrieved")
                    
                    response = result.get("response", "")
                    if response:
                        print(f"✓ LLM Response generated ({len(response)} chars)")
                        print(f"  Preview: {response[:100]}...")
                    else:
                        print("✗ No response generated")
                else:
                    print(f"✗ Query failed: {resp.status}")
        except Exception as e:
            print(f"✗ Query error: {e}")
        
        print("\n" + "="*80)
        print("TEST COMPLETED")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(test_upload_flow())

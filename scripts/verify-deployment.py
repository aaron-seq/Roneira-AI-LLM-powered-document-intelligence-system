#!/usr/bin/env python3
# ==============================================================================
# Deployment Verification Script
# Tests all endpoints and services for the free deployment
# ==============================================================================

import asyncio
import aiohttp
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import time

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "info"):
    """Print colored status messages"""
    color = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }.get(status, Colors.BLUE)
    
    print(f"{color}[{status.upper()}]{Colors.END} {message}")

class DeploymentVerifier:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint(self, endpoint: str, method: str = "GET", 
                          data: Any = None, files: Any = None,
                          expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                async with self.session.get(url) as resp:
                    status = resp.status
                    try:
                        response_data = await resp.json()
                    except:
                        response_data = await resp.text()
            
            elif method == "POST":
                if files:
                    async with self.session.post(url, data=files) as resp:
                        status = resp.status
                        response_data = await resp.json()
                else:
                    async with self.session.post(url, json=data) as resp:
                        status = resp.status
                        response_data = await resp.json()
            
            success = status == expected_status
            return {
                "success": success,
                "status_code": status,
                "response": response_data,
                "url": url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print_status("Testing health check endpoint...", "info")
        
        result = await self.test_endpoint("/health")
        
        if result["success"]:
            health_data = result["response"]
            print_status(f"Health check passed: {health_data.get('status', 'unknown')}", "success")
            
            # Check individual services
            services = health_data.get("services", {})
            for service_name, service_status in services.items():
                status_text = service_status.get("status", "unknown")
                if status_text == "healthy":
                    print_status(f"  ✓ {service_name}: {status_text}", "success")
                else:
                    print_status(f"  ✗ {service_name}: {status_text}", "warning")
            
            self.test_results.append(("Health Check", True, "All services checked"))
            return True
        else:
            print_status(f"Health check failed: {result.get('error', 'Unknown error')}", "error")
            self.test_results.append(("Health Check", False, result.get('error', 'Failed')))
            return False
    
    async def test_file_upload(self) -> str:
        """Test file upload with a sample file"""
        print_status("Testing file upload...", "info")
        
        # Create a simple test file
        test_content = "This is a test document for Roneira AI.\nIt contains sample text for processing."
        
        # Prepare multipart data
        data = aiohttp.FormData()
        data.add_field('file', 
                      test_content,
                      filename='test_document.txt',
                      content_type='text/plain')
        
        try:
            async with self.session.post(f"{self.base_url}/upload", data=data) as resp:
                if resp.status == 200:
                    response_data = await resp.json()
                    document_id = response_data.get("document_id")
                    print_status(f"File upload successful: {document_id}", "success")
                    self.test_results.append(("File Upload", True, f"Document ID: {document_id}"))
                    return document_id
                else:
                    error_text = await resp.text()
                    print_status(f"File upload failed: {resp.status} - {error_text}", "error")
                    self.test_results.append(("File Upload", False, f"Status: {resp.status}"))
                    return None
        
        except Exception as e:
            print_status(f"File upload error: {e}", "error")
            self.test_results.append(("File Upload", False, str(e)))
            return None
    
    async def test_processing_status(self, document_id: str) -> bool:
        """Test processing status endpoint"""
        if not document_id:
            return False
        
        print_status(f"Testing processing status for {document_id}...", "info")
        
        # Poll status for up to 60 seconds
        for i in range(30):  # 30 attempts, 2 seconds each
            result = await self.test_endpoint(f"/documents/{document_id}/status")
            
            if result["success"]:
                status_data = result["response"]
                status = status_data.get("status")
                progress = status_data.get("progress", 0)
                
                print_status(f"  Status: {status} ({progress}%)", "info")
                
                if status == "completed":
                    print_status("Document processing completed!", "success")
                    self.test_results.append(("Processing Status", True, "Completed successfully"))
                    return True
                elif status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    print_status(f"Document processing failed: {error_msg}", "error")
                    self.test_results.append(("Processing Status", False, error_msg))
                    return False
                
                # Wait before next poll
                await asyncio.sleep(2)
            else:
                print_status(f"Status check failed: {result.get('error')}", "error")
                break
        
        print_status("Processing status check timed out", "warning")
        self.test_results.append(("Processing Status", False, "Timed out"))
        return False
    
    async def test_document_analysis(self, document_id: str) -> bool:
        """Test document analysis retrieval"""
        if not document_id:
            return False
        
        print_status(f"Testing document analysis for {document_id}...", "info")
        
        result = await self.test_endpoint(f"/documents/{document_id}")
        
        if result["success"]:
            analysis_data = result["response"]
            text_length = len(analysis_data.get("text", ""))
            confidence = analysis_data.get("confidence", 0)
            processing_time = analysis_data.get("processing_time", 0)
            
            print_status(f"Analysis retrieved: {text_length} chars, {confidence:.1f}% confidence, {processing_time:.2f}s", "success")
            self.test_results.append(("Document Analysis", True, f"Text: {text_length} chars"))
            return True
        else:
            print_status(f"Analysis retrieval failed: {result.get('error')}", "error")
            self.test_results.append(("Document Analysis", False, result.get('error', 'Failed')))
            return False
    
    async def test_document_listing(self) -> bool:
        """Test document listing endpoint"""
        print_status("Testing document listing...", "info")
        
        result = await self.test_endpoint("/documents")
        
        if result["success"]:
            documents = result["response"]
            doc_count = len(documents) if isinstance(documents, list) else 0
            print_status(f"Document listing successful: {doc_count} documents", "success")
            self.test_results.append(("Document Listing", True, f"{doc_count} documents"))
            return True
        else:
            print_status(f"Document listing failed: {result.get('error')}", "error")
            self.test_results.append(("Document Listing", False, result.get('error', 'Failed')))
            return False
    
    def print_summary(self):
        """Print test summary"""
        print_status("\n" + "="*60, "info")
        print_status("DEPLOYMENT VERIFICATION SUMMARY", "info")
        print_status("="*60, "info")
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, details in self.test_results:
            status = "✓" if success else "✗"
            color = "success" if success else "error"
            print_status(f"{status} {test_name}: {details}", color)
        
        print_status(f"\nResults: {passed}/{total} tests passed", 
                    "success" if passed == total else "warning")
        
        if passed == total:
            print_status("🎉 All tests passed! Your deployment is working correctly.", "success")
        else:
            print_status("⚠️  Some tests failed. Check the logs above for details.", "warning")
        
        return passed == total

async def main():
    """Main verification function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Roneira AI deployment")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the deployment (default: http://localhost:8000)")
    args = parser.parse_args()
    
    print_status(f"Starting deployment verification for: {args.url}", "info")
    print_status("="*60 + "\n", "info")
    
    async with DeploymentVerifier(args.url) as verifier:
        # Test health check
        health_ok = await verifier.test_health_check()
        
        if not health_ok:
            print_status("Health check failed. Stopping verification.", "error")
            verifier.print_summary()
            sys.exit(1)
        
        # Test file upload
        document_id = await verifier.test_file_upload()
        
        if document_id:
            # Test processing status
            processing_ok = await verifier.test_processing_status(document_id)
            
            if processing_ok:
                # Test document analysis
                await verifier.test_document_analysis(document_id)
        
        # Test document listing
        await verifier.test_document_listing()
        
        # Print summary
        success = verifier.print_summary()
        
        if success:
            print_status("\n🚀 Your Roneira AI deployment is fully functional!", "success")
            sys.exit(0)
        else:
            print_status("\n❌ Deployment verification failed.", "error")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
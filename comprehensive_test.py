"""
Comprehensive Test Suite for Roneira Document Intelligence System
Covers: API, RAG, Edge Cases, Error Handling
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"
RESULTS = []


def log_result(test_name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append({"test": test_name, "passed": passed, "details": details})
    print(f"{status} | {test_name}")
    if details and not passed:
        print(f"       └─ {details[:100]}")


# ==================== API TESTS ====================


def test_health():
    """Test /health endpoint"""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        log_result("API Health Check", r.status_code == 200, f"Status: {r.status_code}")
    except Exception as e:
        log_result("API Health Check", False, str(e))


def test_documents_list():
    """Test /documents endpoint"""
    try:
        r = requests.get(f"{BASE_URL}/documents", timeout=5)
        data = r.json()
        log_result(
            "Document List",
            r.status_code == 200 and isinstance(data, list),
            f"Count: {len(data)}",
        )
        return data
    except Exception as e:
        log_result("Document List", False, str(e))
        return []


def test_document_download(doc_id):
    """Test /documents/{id}/download endpoint"""
    try:
        r = requests.get(f"{BASE_URL}/documents/{doc_id}/download", timeout=10)
        log_result(
            "Document Download", r.status_code == 200, f"Size: {len(r.content)} bytes"
        )
    except Exception as e:
        log_result("Document Download", False, str(e))


# ==================== RAG TESTS ====================


def test_query(query, expected_sources=True, description=""):
    """Test /query endpoint with specific query"""
    try:
        r = requests.post(
            f"{BASE_URL}/query", json={"query": query, "detailed": False}, timeout=60
        )
        data = r.json()
        has_sources = len(data.get("sources", [])) > 0
        passed = (expected_sources and has_sources) or (
            not expected_sources and not has_sources
        )
        log_result(
            f"RAG: {description}",
            passed,
            f"Sources: {len(data.get('sources', []))}, Response: {data.get('response', '')[:50]}...",
        )
        return data
    except Exception as e:
        log_result(f"RAG: {description}", False, str(e))
        return {}


# ==================== ERROR HANDLING TESTS ====================


def test_invalid_document_id():
    """Test error handling for invalid document ID"""
    try:
        r = requests.get(f"{BASE_URL}/documents/invalid-uuid-12345/download", timeout=5)
        log_result(
            "Invalid Document ID", r.status_code == 404, f"Status: {r.status_code}"
        )
    except Exception as e:
        log_result("Invalid Document ID", False, str(e))


def test_malformed_query():
    """Test error handling for malformed query request"""
    try:
        r = requests.post(
            f"{BASE_URL}/query", json={"invalid_field": "test"}, timeout=5
        )
        log_result(
            "Malformed Query Request", r.status_code == 422, f"Status: {r.status_code}"
        )
    except Exception as e:
        log_result("Malformed Query Request", False, str(e))


# ==================== MAIN ====================


def run_all_tests():
    print("\n" + "=" * 60)
    print("RONEIRA DOCUMENT INTELLIGENCE SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60 + "\n")

    # API Tests
    print("\n--- API & Backend Tests ---")
    test_health()
    docs = test_documents_list()
    if docs:
        test_document_download(docs[0].get("id", ""))

    # RAG Tests
    print("\n--- RAG & LLM Tests ---")
    test_query(
        "What is the paid time off policy?",
        expected_sources=True,
        description="HR Policy Query",
    )
    test_query(
        "What are the invoice details?",
        expected_sources=True,
        description="Invoice Query",
    )
    test_query(
        "What are the engineering specifications?",
        expected_sources=True,
        description="Engineering Query",
    )

    # Edge Case Tests
    print("\n--- Edge Case Tests ---")
    test_query("How are you?", expected_sources=False, description="Chitchat Handling")
    test_query("Hello", expected_sources=False, description="Greeting Handling")
    test_query(
        "What is the address of Enterprise Solutions LLC?",
        expected_sources=False,
        description="Missing Information",
    )
    test_query("", expected_sources=False, description="Empty Query")

    # Error Handling Tests
    print("\n--- Error Handling Tests ---")
    test_invalid_document_id()
    test_malformed_query()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)
    print(f"RESULTS: {passed}/{total} tests passed ({100 * passed // total}%)")
    print("=" * 60 + "\n")

    # Save results
    with open("test_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("Results saved to test_results.json")


if __name__ == "__main__":
    run_all_tests()

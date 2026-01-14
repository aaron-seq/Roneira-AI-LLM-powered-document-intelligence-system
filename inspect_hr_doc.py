import requests
import sys

try:
    response = requests.get("http://localhost:8000/documents")
    response.raise_for_status()
    docs = response.json()

    print(f"Total documents: {len(docs)}")

    # Use 'filename' as per API response
    hr_docs = [d for d in docs if "hr" in d.get("filename", "").lower()]
    print(f"Found {len(hr_docs)} HR docs")

    for doc in hr_docs:
        doc_id = doc["id"]
        filename = doc["filename"]
        print(f"\n--- {filename} (ID: {doc_id}) ---")

        # Fetch details
        detail_resp = requests.get(f"http://localhost:8000/documents/{doc_id}")
        if detail_resp.status_code == 200:
            detail = detail_resp.json()
            text = detail.get("text", "")

            if text:
                print(f"Text length: {len(text)}")
                print(f"First 500 chars:\n{text[:500]}")

                # Check for keywords
                keywords = ["paid", "time", "off", "policy", "pto", "leave"]
                print("\nKeyword checks:")
                text_lower = text.lower()
                for kw in keywords:
                    count = text_lower.count(kw)
                    print(f"  '{kw}': {count}")
            else:
                print("extracted_text is empty or None in detail view")
        else:
            print(f"Failed to fetch details: {detail_resp.status_code}")

except Exception as e:
    print(f"Error: {e}")

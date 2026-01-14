import requests
import time
import json

BASE_URL = "http://localhost:8000"


def log(msg):
    print(msg)
    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def test_query(
    query, expected_sources_count=None, should_have_sources=True, description=""
):
    log(f"\nTesting: {description} ('{query}')")
    try:
        response = requests.post(
            f"{BASE_URL}/query", json={"query": query, "detailed": False}
        )
        response.raise_for_status()
        data = response.json()

        log(f"Response: {data['response'][:100]}...")
        log(f"Sources: {len(data['sources'])}")

        if should_have_sources:
            if len(data["sources"]) == 0:
                log("FAILED: Expected sources, found none.")
                return False
            if (
                expected_sources_count is not None
                and len(data["sources"]) != expected_sources_count
            ):
                log(
                    f"WARNING: Expected {expected_sources_count} sources, found {len(data['sources'])}."
                )
        else:
            if len(data["sources"]) > 0:
                log("FAILED: Expected NO sources, found some.")
                log(f"First source: {data['sources'][0]}")
                return False

        log("PASSED")
        return True
    except Exception as e:
        log(f"ERROR: {e}")
        return False


def check_health():
    try:
        requests.get(f"{BASE_URL}/health")
        return True
    except:
        return False


def main():
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("Starting Verification\n")

    log("Waiting for API...")
    for _ in range(10):
        if check_health():
            break
        time.sleep(2)

    results = []

    # Test 1: Chitchat
    results.append(
        test_query(
            "How are you?", should_have_sources=False, description="Chitchat Greeting"
        )
    )

    # Test 2: Missing Info
    results.append(
        test_query(
            "What is the address of Enterprise Solutions LLC?",
            should_have_sources=False,
            description="Missing Information",
        )
    )

    # Test 3: Valid Query
    results.append(
        test_query(
            "What is the paid time off policy?",
            should_have_sources=True,
            description="Valid Document Query",
        )
    )

    if all(results):
        log("\nALL EDGE CASE TESTS PASSED")
    else:
        log("\nSOME TESTS FAILED")


if __name__ == "__main__":
    main()

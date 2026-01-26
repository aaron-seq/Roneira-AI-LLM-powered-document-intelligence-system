# Testing Guidelines & Best Practices

## Table of Contents
1. [Core Methodology](#core-methodology)
2. [Validation Framework](#validation-framework)
3. [Before & After State Verification](#before--after-state-verification)
4. [Tooling & Infrastructure](#tooling--infrastructure)
5. [E2E Testing Goals](#e2e-testing-goals)
6. [Best Practices](#best-practices)

---

## Core Methodology

Our testing strategy is built on **objective validation** rather than subjective passing. We employ a rigorous **7-Step Validation Process** to ensure every artifact and functionality is robust.

1.  **Start with Requirements**: Extract explicit constraints; do not guess.
2.  **Use Domain Expertise**: Model real production data patterns.
3.  **Balance Positive/Negative**: Validate success paths and error handling equally.
4.  **Be Specific**: Use exact values (e.g., "$380.03") instead of generic checks ("value changed").
5.  **Automate**: Leverage Faker, Hypothesis, and Mutation testing.
6.  **Manual Curation**: Review generated data for realism.
7.  **Document Methodology**: Ensure reproducibility.

---

## Validation Framework

We use a specific template for creating comprehensive test cases:

```python
TEST_CASE_ID = "TC_001_ORDER_PLACEMENT"
ARTIFACT = "OrderService.place_order"
OBJECTIVE = "Verify valid order placement reduces inventory and deducts balance"

# 1. State Documentation Requirements
context "Before State" {
    user_balance = 500.00
    inventory_count = 100
    order_queue_size = 0
}

# 2. Action Execution
action_trigger = "POST /api/orders/place"
payload = {"item_id": "SKU-001", "quantity": 2, "price": 110.00}

# 3. Expected State (Calculated)
expected_balance = 500.00 - (110.00 * 2) # = 280.00
expected_inventory = 100 - 2 # = 98

# 4. Validation Assertions
assert user.balance == expected_balance
assert inventory.count == expected_inventory
assert order_queue_size == 1

# 5. Comparison Assertion Framework
# Use exact matching for critical financial data
# Use range-validation for timestamps (within 2s)
```

---

## Before & After State Verification

Every significant test MUST document the **Before** and **After** states explicitly.

### 1. State Documentation Requirements
-   **Current State (Before)**: Snapshot of initial conditions (DB, Cache, UI).
-   **Expected State (After)**: Predicted outcomes including side effects.
-   **Validation Criteria**: Measurable assertions.

### 2. Baseline Establishment
-   Capture pre-execution snapshots.
-   Log exact inputs.

### 3. Execution & Isolation
-   Ensure no external interference.
-   Log timing and intermediate states.

### 4. Post-Execution Verification
-   **Field-level comparison**: Validate every changed field.
-   **Side Effect Detection**: Checks DB rows, cache keys, external service calls.
-   **Invariant Validation**: Confirm strictly what *should not* change.

---

## Tooling & Infrastructure

### Backend
-   **Pytest**: Main test runner.
-   **Faker**: Generating realistic user data (emails, names, addresses).
-   **Hypothesis**: Property-based testing to find edge cases.
    -   Example: `@given(text(min_size=1, max_size=100))`
-   **Mutation Testing (Mutmut)**: modifying code to ensure tests catch bugs.

### Frontend / E2E
-   **Cypress**: Dashboard testing, video feed validation.
-   **Playwright**: Cross-browser testing (Webkit, Chromium, Firefox).
-   **Selenium**: Integration testing complex workflows.

---

## E2E Testing Goals

-   **Accuracy Scores**: Measure system performance.
-   **Confusion Matrix**: Use `scikit-learn` to calculate distinct validation metrics for AI/ML components.
-   **Visual Regression**: Screenshot comparison for UI components (Before/After).

---

## Best Practices

1.  **Production Parity**: Test in environments closely matching production.
2.  **Idempotency**: Verify retriable operations do not duplicate side effects.
3.  **Observability**: Validate logs and metrics are emitted during tests.
4.  **Error Messages**: Assert exact error message content, not just "error occurred".

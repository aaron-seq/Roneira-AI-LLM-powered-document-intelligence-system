import pytest
from faker import Faker
from hypothesis import given, strategies as st
from dataclasses import dataclass, field
from typing import List, Optional

# --- Mock Artifact / System Under Test ---
@dataclass
class OrderSystem:
    balance: float = 500.00
    inventory: dict = field(default_factory=lambda: {"SKU-001": 100})
    orders: List[dict] = field(default_factory=list)

    def place_order(self, sku: str, quantity: int, price: float):
        if sku not in self.inventory:
            raise ValueError("Item not found")
        if self.inventory[sku] < quantity:
            raise ValueError("Insufficient stock")
        if self.balance < (quantity * price):
            raise ValueError("Insufficient funds")
        
        # State transitions
        self.inventory[sku] -= quantity
        cost = quantity * price
        self.balance -= cost
        self.orders.append({
            "sku": sku,
            "quantity": quantity,
            "cost": cost,
            "status": "confirmed"
        })
        return {"status": "success", "remaining_balance": self.balance}

# --- Testing Implementation ---

fake = Faker()

class TestOrderSystemValidation:
    
    def test_order_placement_state_transition(self):
        """
        Validates the 'Before' and 'After' state of an order placement
        using the strict validation methodology.
        """
        # 1. Document Requirements & Context (Before State)
        system = OrderSystem()
        initial_balance = system.balance # 500.00
        initial_stock = system.inventory["SKU-001"] # 100
        initial_order_count = len(system.orders) # 0
        
        # Action Inputs
        sku = "SKU-001"
        qty = 2
        price = 50.00
        
        # 2. Expected State Derivation (The Math)
        expected_cost = qty * price # 100.00
        expected_balance = initial_balance - expected_cost # 400.00
        expected_stock = initial_stock - qty # 98
        expected_order_count = initial_order_count + 1
        
        # 3. Execution
        result = system.place_order(sku=sku, quantity=qty, price=price)
        
        # 4. Post-Execution Verification (Assertions)
        
        # A. Return Value check
        assert result["status"] == "success"
        assert result["remaining_balance"] == expected_balance
        
        # B. Side Effect: Balance Update
        assert system.balance == expected_balance, \
            f"Balance mismatch! Expected {expected_balance}, got {system.balance}"
            
        # C. Side Effect: Inventory Update
        assert system.inventory[sku] == expected_stock, \
            f"Stock mismatch! Expected {expected_stock}, got {system.inventory[sku]}"
            
        # D. Side Effect: Order Log
        assert len(system.orders) == expected_order_count
        last_order = system.orders[-1]
        assert last_order["sku"] == sku
        assert last_order["status"] == "confirmed"

    @given(st.integers(min_value=1, max_value=50))
    def test_property_based_stock_reduction(self, quantity):
        """
        Hypothesis test to verify stock logic holds for any valid integer input.
        """
        system = OrderSystem()
        price = 1.00 # inexpensive to avoid balance issues
        
        initial_stock = system.inventory["SKU-001"]
        
        # Assuming we only test valid quantities that fit in stock/budget for this property test
        if quantity <= initial_stock and (quantity * price) <= system.balance:
            system.place_order("SKU-001", quantity, price)
            assert system.inventory["SKU-001"] == initial_stock - quantity
        else:
            with pytest.raises(ValueError):
                system.place_order("SKU-001", quantity, price)


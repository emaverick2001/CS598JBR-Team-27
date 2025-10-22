import pytest
from HumanEval_3 import below_zero

def check_balance(operations):
    balance = 0

    for op in operations:
        balance += op
        if balance < 0:
            return True

    return False

def test_empty_input():
    assert below_zero([]) == False

def test_zero():
    assert below_zero([0]) == False

def test_negative():
    assert below_zero([-1]) == True

def test_none():
    with pytest.raises(TypeError):
        below_zero(None)

def test_boundary_conditions():
    assert below_zero([1000000]) == False

def test_normal_cases():
    assert below_zero([1, 2, 3]) == False

def test_edge_cases():
    assert below_zero([1, -2, 3]) == True

def test_multiple_operations():
    assert below_zero([1, -2, 3, 4, -5, 6]) == True

def test_large_numbers():
    assert below_zero([1000000, -2000000, 3000000]) == False

def test_negative_numbers():
    assert below_zero([-1, -2, -3]) == True

def test_zero_balance():
    assert below_zero([1, -1]) == False

def test_negative_balance():
    assert below_zero([1, -2, -3]) == True

def test_positive_balance():
    assert below_zero([1, 2, 3]) == False

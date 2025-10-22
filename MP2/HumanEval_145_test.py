import pytest
from HumanEval_145 import order_by_points

def test_empty_input():
    assert order_by_points([]) == []

def test_single_digit():
    assert order_by_points([1]) == [1]

def test_negative_numbers():
    assert order_by_points([-1, -2, -3]) == [-1, -2, -3]

def test_zero():
    assert order_by_points([0, 0, 0]) == [0, 0, 0]

def test_positive_numbers():
    assert order_by_points([1, 2, 3]) == [1, 2, 3]

def test_positive_numbers_with_different_digits():
    assert order_by_points([12, 23, 34]) == [12, 23, 34]

def test_positive_numbers_with_same_digits():
    assert order_by_points([11, 22, 33]) == [11, 22, 33]

def test_mixed_positive_and_negative_numbers():
    assert order_by_points([1, -2, 3]) == [-2, 1, 3]

def test_boundary_conditions():
    assert order_by_points([10**9, -10**9, 0]) == [-10**9, 0, 10**9]

def test_large_numbers():
    assert order_by_points([1234567890, 9876543210, 5555555555]) == [1234567890, 5555555555, 9876543210]

def test_none_input():
    with pytest.raises(TypeError):
        order_by_points(None)

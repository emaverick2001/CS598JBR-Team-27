import pytest
from HumanEval_109 import move_one_ball

def test_empty_input():
    assert move_one_ball([]) == True

def test_single_element():
    assert move_one_ball([5]) == True

def test_sorted_array():
    assert move_one_ball([1, 2, 3, 4, 5]) == True

def test_reversed_array():
    assert move_one_ball([5, 4, 3, 2, 1]) == True

def test_negative_numbers():
    assert move_one_ball([-5, -4, -3, -2, -1]) == True

def test_zero():
    assert move_one_ball([0, 0, 0, 0]) == True

def test_negative_positive_mixed():
    assert move_one_ball([-5, -4, 3, 2, 1]) == True

def test_duplicates():
    assert move_one_ball([1, 2, 2, 1]) == True

def test_non_rotation():
    assert move_one_ball([1, 2, 3, 4]) == False

def test_large_input():
    large_array = list(range(1, 10**6 + 1))
    assert move_one_ball(large_array) == True

def test_large_input_reversed():
    large_array = list(range(10**6, 0, -1))
    assert move_one_ball(large_array) == True

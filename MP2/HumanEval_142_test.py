import pytest
from HumanEval_142 import sum_squares

def test_empty_input():
    assert sum_squares([]) == 0

def test_zero_input():
    assert sum_squares([0, 0, 0, 0]) == 0

def test_negative_input():
    assert sum_squares([-1, -2, -3, -4]) == -10

def test_none_input():
    assert sum_squares([None, None, None, None]) == 0

def test_normal_cases():
    assert sum_squares([1, 2, 3, 4]) == 11
    assert sum_squares([2, 3, 4, 5]) == 28
    assert sum_squares([3, 4, 5, 6]) == 46

def test_edge_cases():
    assert sum_squares([10, 10, 10, 10]) == 100
    assert sum_squares([20, 20, 20, 20]) == 800
    assert sum_squares([30, 30, 30, 30]) == 2700

def test_boundary_conditions():
    assert sum_squares([100, 100, 100, 100]) == 10000
    assert sum_squares([200, 200, 200, 200]) == 160000
    assert sum_squares([300, 300, 300, 300]) == 270000

def test_all_execution_paths_and_branches():
    assert sum_squares([1, 2, 3, 1]) == 5
    assert sum_squares([2, 3, 4, 2]) == 15
    assert sum_squares([3, 4, 5, 3]) == 33
    assert sum_squares([1, 2, 3, 4]) == 11
    assert sum_squares([2, 3, 4, 5]) == 28
    assert sum_squares([3, 4, 5, 6]) == 46
    assert sum_squares([1, 2, 3, 2]) == 9
    assert sum_squares([2, 3, 4, 3]) == 27
    assert sum_squares([3, 4, 5, 4]) == 64

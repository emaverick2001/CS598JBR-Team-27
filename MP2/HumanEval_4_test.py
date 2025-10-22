import pytest
from HumanEval_4 import mean_absolute_deviation

def test_empty_input():
    assert mean_absolute_deviation([]) is None

def test_single_element():
    assert mean_absolute_deviation([5]) == 0

def test_negative_numbers():
    assert mean_absolute_deviation([-1, -2, -3]) == 2

def test_zero():
    assert mean_absolute_deviation([0, 0, 0]) == 0

def test_positive_numbers():
    assert mean_absolute_deviation([1, 2, 3]) == 1

def test_mixed_numbers():
    assert mean_absolute_deviation([1, -2, 3]) == 2

def test_large_numbers():
    assert mean_absolute_deviation([1e6, 2e6, 3e6]) == 1e6

def test_small_numbers():
    assert mean_absolute_deviation([1e-6, 2e-6, 3e-6]) == 1e-6

def test_none_input():
    assert mean_absolute_deviation(None) is None

def test_non_numeric_input():
    with pytest.raises(TypeError):
        mean_absolute_deviation(['a', 'b', 'c'])

def test_large_input():
    numbers = list(range(1, 1001))
    assert mean_absolute_deviation(numbers) == 500.5

def test_small_input():
    numbers = list(range(1, 11))
    assert mean_absolute_deviation(numbers) == 3.3027756377319946

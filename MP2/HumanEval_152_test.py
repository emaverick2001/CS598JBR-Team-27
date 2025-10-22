import pytest
from HumanEval_152 import compare

def test_empty_inputs():
    assert compare([], []) == []

def test_zero_values():
    assert compare([0, 0, 0], [0, 0, 0]) == [0, 0, 0]

def test_negative_numbers():
    assert compare([-1, -2, -3], [1, 2, 3]) == [2, 4, 6]

def test_none_values():
    with pytest.raises(TypeError):
        compare(None, None)

def test_boundary_conditions():
    assert compare([10, 10, 10], [0, 5, 10]) == [10, 5, 10]

def test_normal_cases():
    assert compare([5, 10, 15], [0, 5, 10]) == [5, 5, 15]

def test_unequal_lengths():
    assert compare([1, 2, 3], [1, 2]) == [0, 1]

def test_single_element():
    assert compare([1], [1]) == [0]

def test_large_inputs():
    game = list(range(1, 1001))
    guess = list(range(1001, 2001))
    assert compare(game, guess) == [1000]*1000

def test_large_negative_inputs():
    game = list(range(-1000, 0))
    guess = list(range(-999, 0))
    assert compare(game, guess) == [1]*1000

def test_mixed_negative_and_positive_inputs():
    game = [-1, 2, -3, 4]
    guess = [1, -2, 3, -4]
    assert compare(game, guess) == [2, 5, 7, 8]

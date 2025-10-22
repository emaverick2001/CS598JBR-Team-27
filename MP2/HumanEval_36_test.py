import pytest
from HumanEval_36 import fizz_buzz

def test_empty_input():
    assert fizz_buzz(0) == 0

def test_negative_input():
    assert fizz_buzz(-1) == 0

def test_zero():
    assert fizz_buzz(0) == 0

def test_small_input():
    assert fizz_buzz(10) == 1  # '7' appears once in 0, 7, 10

def test_large_input():
    assert fizz_buzz(100) == 5  # '7' appears five times in 0, 7, 10, 17, 47, 70, 77, 84, 91, 98

def test_boundary_input():
    assert fizz_buzz(1000) == 11  # '7' appears eleven times in numbers that are multiples of 11 or 13

def test_normal_input():
    assert fizz_buzz(10000) == 153  # '7' appears 153 times in numbers that are multiples of 11 or 13

def test_none_input():
    with pytest.raises(TypeError):
        fizz_buzz(None)

def test_string_input():
    with pytest.raises(TypeError):
        fizz_buzz('abc')

def test_list_input():
    with pytest.raises(TypeError):
        fizz_buzz([1, 2, 3])

def test_dict_input():
    with pytest.raises(TypeError):
        fizz_buzz({'a': 1, 'b': 2})

def test_float_input():
    with pytest.raises(TypeError):
        fizz_buzz(1.23)

def test_negative_float_input():
    assert fizz_buzz(-1.23) == 0

def test_negative_large_input():
    assert fizz_buzz(-10000) == 0

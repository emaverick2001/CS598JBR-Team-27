import pytest
from HumanEval_92 import any_int

def test_valid_inputs():
    assert any_int(3, 4, 5) == True
    assert any_int(5, 12, 13) == True
    assert any_int(6, 8, 10) == True
    assert any_int(7, 24, 25) == True
    assert any_int(8, 15, 17) == True
    assert any_int(9, 21, 25) == True
    assert any_int(10, 24, 26) == True
    assert any_int(12, 15, 20) == True
    assert any_int(13, 26, 29) == True
    assert any_int(15, 20, 25) == True

def test_invalid_inputs():
    assert any_int(1, 2, 3) == False
    assert any_int(2, 3, 4) == False
    assert any_int(3, 4, 6) == False
    assert any_int(3, 5, 7) == False
    assert any_int(4, 5, 9) == False
    assert any_int(4, 6, 8) == False
    assert any_int(5, 6, 9) == False
    assert any_int(5, 7, 10) == False
    assert any_int(6, 7, 11) == False
    assert any_int(6, 8, 12) == False

def test_non_integer_inputs():
    assert any_int(1.5, 2, 3) == False
    assert any_int(2, 3.5, 4) == False
    assert any_int(3, 4, 6.5) == False
    assert any_int(3.5, 5, 7) == False
    assert any_int(4, 5.5, 9) == False
    assert any_int(4.5, 6, 8) == False
    assert any_int(5, 6.5, 9) == False
    assert any_int(5.5, 7, 10) == False
    assert any_int(6, 7.5, 11) == False
    assert any_int(6.5, 8, 12) == False

def test_negative_inputs():
    assert any_int(-1, 2, 3) == False
    assert any_int(1, -2, 3) == False
    assert any_int(1, 2, -3) == False
    assert any_int(-1, -2, 3) == False
    assert any_int(-1, 2, -3) == False
    assert any_int(1, -2, -3) == False
    assert any_int(-1, -2, -3) == False

def test_zero_inputs():
    assert any_int(0, 2, 3) == False
    assert any_int(1, 0, 3) == False
    assert any_int(1, 2, 0) == False
    assert any_int(0, 0, 3) == False
    assert any_int(0, 2,

import pytest
from HumanEval_97 import multiply

def test_positive_positive():
    assert abs(15 % 10) * abs(20 % 10) == 100

def test_positive_negative():
    assert abs(15 % 10) * abs(-20 % 10) == 100

def test_negative_positive():
    assert abs(-15 % 10) * abs(20 % 10) == 100

def test_negative_negative():
    assert abs(-15 % 10) * abs(-20 % 10) == 100

def test_zero_positive():
    assert abs(0 % 10) * abs(20 % 10) == 0

def test_positive_zero():
    assert abs(15 % 10) * abs(0 % 10) == 0

def test_zero_zero():
    assert abs(0 % 10) * abs(0 % 10) == 0

def test_zero_negative():
    assert abs(0 % 10) * abs(-20 % 10) == 0

def test_negative_zero():
    assert abs(-15 % 10) * abs(0 % 10) == 0

def test_large_positive():
    assert abs(1234567890 % 10) * abs(9876543210 % 10) == 9671406440

def test_large_negative():
    assert abs(-1234567890 % 10) * abs(9876543210 % 10) == 9671406440

def test_negative_large():
    assert abs(-1234567890 % 10) * abs(-9876543210 % 10) == 9671406440

def test_large_zero():
    assert abs(1234567890 % 10) * abs(0 % 10) == 0

def test_zero_large():
    assert abs(0 % 10) * abs(9876543210 % 10) == 0

def test_large_large():
    assert abs(1234567890 % 10) * abs(9876543210 % 10) == 9671406440

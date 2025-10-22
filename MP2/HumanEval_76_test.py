import pytest
from HumanEval_76 import is_simple_power

def is_power_of_n(n, x):
    if (n == 1):
        return (x == 1)
    power = 1
    while (power < x):
        power = power * n
    return (power == x)

def test_is_power_of_n():
    assert is_simple_power(2, 4) == True
    assert is_simple_power(2, 5) == False
    assert is_simple_power(3, 9) == True
    assert is_simple_power(3, 8) == False
    assert is_simple_power(1, 1) == True
    assert is_simple_power(1, 2) == False
    assert is_simple_power(0, 0) == True
    assert is_simple_power(0, 1) == False
    assert is_simple_power(2, 0) == True
    assert is_simple_power(2, -4) == False
    assert is_simple_power(-2, 4) == False
    assert is_simple_power(-2, -4) == True
    assert is_simple_power(None, None) == True
    assert is_simple_power(None, 0) == False
    assert is_simple_power(2, None) == False

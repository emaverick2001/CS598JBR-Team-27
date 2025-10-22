import pytest
from HumanEval_112 import reverse_delete

def test_empty_input():
    s = ""
    c = ""
    result = your_function(s, c)
    assert result == ("", True)

def test_no_char_removed():
    s = "hello"
    c = "world"
    result = your_function(s, c)
    assert result == ("hello", False)

def test_all_char_removed():
    s = "hello"
    c = "helo"
    result = your_function(s, c)
    assert result == ("", True)

def test_one_char_removed():
    s = "hello"
    c = "eo"
    result = your_function(s, c)
    assert result == ("hl", False)

def test_zero():
    s = "hello"
    c = "0"
    result = your_function(s, c)
    assert result == ("hello", False)

def test_negative_numbers():
    s = "hello"
    c = "-1"
    result = your_function(s, c)
    assert result == ("hello", False)

def test_none():
    s = "hello"
    c = None
    result = your_function(s, c)
    assert result == ("hello", False)

def test_boundary_conditions():
    s = "a" * 1000
    c = "a" * 999
    result = your_function(s, c)
    assert result == ("a", True)

def test_normal_cases():
    s = "hello"
    c = "eo"
    result = your_function(s, c)
    assert result == ("hl", False)

def test_large_input():
    s = "a" * 1000000
    c = "a" * 500000
    result = your_function(s, c)
    assert result == ("a" * 500000, True)

def test_large_input_no_char_removed():
    s = "a" * 1000000
    c = ""
    result = your_function(s, c)
    assert result == ("a" * 1000000, True)

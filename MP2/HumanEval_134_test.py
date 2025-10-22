import pytest
from HumanEval_134 import check_if_last_char_is_a_letter

def check_last_char(txt):
    check = txt.split(' ')[-1]
    return True if len(check) == 1 and (97 <= ord(check.lower()) <= 122) else False

def test_empty_input():
    assert check_if_last_char_is_a_letter("") == False

def test_zero():
    assert check_if_last_char_is_a_letter("0") == False

def test_negative_numbers():
    assert check_if_last_char_is_a_letter("-1") == False

def test_none():
    assert check_if_last_char_is_a_letter(None) == False

def test_uppercase_letters():
    assert check_if_last_char_is_a_letter("HELLO WORLD") == False

def test_lowercase_letters():
    assert check_if_last_char_is_a_letter("hello world") == True

def test_numbers_in_middle():
    assert check_if_last_char_is_a_letter("hello 123 world") == False

def test_numbers_at_end():
    assert check_if_last_char_is_a_letter("hello world123") == False

def test_special_characters():
    assert check_if_last_char_is_a_letter("hello world!@#") == False

def test_multiple_spaces():
    assert check_if_last_char_is_a_letter("hello  world") == False

def test_boundary_conditions():
    assert check_if_last_char_is_a_letter("a") == True
    assert check_if_last_char_is_a_letter("z") == True
    assert check_if_last_char_is_a_letter("A") == False
    assert check_if_last_char_is_a_letter("Z") == False

def test_normal_cases():
    assert check_if_last_char_is_a_letter("hello world") == False
    assert check_if_last_char_is_a_letter("hello world a") == True
    assert check_if_last_char_is_a_letter("hello world z") == True

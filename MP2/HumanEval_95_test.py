import pytest
from HumanEval_95 import check_dict_case

def test_empty_dict():
    assert check_dict_case({}) == False

def test_single_uppercase_key():
    assert check_dict_case({'KEY': 'value'}) == True

def test_single_lowercase_key():
    assert check_dict_case({'key': 'value'}) == True

def test_mixed_case_keys():
    assert check_dict_case({'Key': 'value'}) == False

def test_multiple_keys():
    assert check_dict_case({'KEY': 'value', 'key': 'value'}) == False

def test_keys_with_numbers():
    assert check_dict_case({'KEY1': 'value', 'key2': 'value'}) == False

def test_keys_with_special_characters():
    assert check_dict_case({'KEY!': 'value', 'key@': 'value'}) == False

def test_keys_with_empty_string():
    assert check_dict_case({'': 'value', 'key': 'value'}) == False

def test_keys_with_none():
    assert check_dict_case({None: 'value', 'key': 'value'}) == False

def test_keys_with_negative_numbers():
    assert check_dict_case({-1: 'value', 'key': 'value'}) == False

def test_keys_with_zero():
    assert check_dict_case({0: 'value', 'key': 'value'}) == False

def test_keys_with_upper_and_lower_case():
    assert check_dict_case({'Key': 'value', 'key': 'value'}) == False

def test_keys_with_upper_and_mixed_case():
    assert check_dict_case({'Key': 'value', 'kEy': 'value'}) == False

def test_keys_with_lower_and_mixed_case():
    assert check_dict_case({'key': 'value', 'kEy': 'value'}) == False

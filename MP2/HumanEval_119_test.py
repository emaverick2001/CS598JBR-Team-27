import pytest
from HumanEval_119 import match_parens

def test_empty_input():
    assert match_parens([]) == 'No'

def test_single_element():
    assert match_parens(['(']) == 'Yes'
    assert match_parens([')']) == 'Yes'

def test_two_elements():
    assert match_parens(['(', ')']) == 'Yes'
    assert match_parens([')', '(']) == 'Yes'
    assert match_parens(['(', '(']) == 'No'
    assert match_parens([')', ')']) == 'Yes'

def test_three_elements():
    assert match_parens(['(', '(', ')']) == 'Yes'
    assert match_parens(['(', ')', '(']) == 'Yes'
    assert match_parens([')', '(', ')']) == 'Yes'
    assert match_parens([')', '(', '(']) == 'No'
    assert match_parens(['(', '(', ')', ')']) == 'Yes'
    assert match_parens(['(', ')', '(', ')']) == 'Yes'
    assert match_parens([')', '(', '(', ')']) == 'Yes'
    assert match_parens([')', ')', '(', '(']) == 'No'

def test_mixed_elements():
    assert match_parens(['(', '(', ')', ')', '(', ')']) == 'Yes'
    assert match_parens(['(', ')', '(', '(', ')', ')']) == 'Yes'
    assert match_parens([')', '(', '(', ')', '(', ')']) == 'Yes'
    assert match_parens([')', ')', '(', '(', ')', '(', ')']) == 'No'

def test_negative_cases():
    assert match_parens(['(', ')', '(', '(', ')', ')', ')']) == 'No'
    assert match_parens(['(', '(', '(', ')', ')', ')', ')']) == 'No'
    assert match_parens([')', '(', '(', '(', ')', ')', ')']) == 'No'
    assert match_parens([')', ')', '(', '(', '(', ')', ')', ')']) == 'No'

def test_boundary_cases():
    assert match_parens(['(', '(', '(', '(', ')', ')', ')', ')']) == 'Yes'
    assert match_parens(['(', '(', '(', ')', ')', '(', ')', ')']) == 'Yes'
    assert match_parens(['(', '(', ')', '(', ')', '(', ')', ')']) == 'Yes'
    assert match_parens(['(', ')', '(', ')', '(', ')', '(', ')']) == 'Yes'

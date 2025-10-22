import pytest
from HumanEval_64 import vowels_count

def count_vowels(s):
    vowels = "aeiouAEIOU"
    n_vowels = sum(c in vowels for c in s)
    if s and (s[-1] == 'y' or s[-1] == 'Y'):
        n_vowels += 1
    return n_vowels

def test_count_vowels():
    assert vowels_count('') == 0
    assert vowels_count('a') == 1
    assert vowels_count('Y') == 1
    assert vowels_count('Hello') == 2
    assert vowels_count('Bye') == 2
    assert vowels_count('Python') == 2
    assert vowels_count('Aeiou') == 5
    assert vowels_count('y') == 1
    assert vowels_count('Yo') == 1
    assert vowels_count('123') == 0
    assert vowels_count('0') == 0
    assert vowels_count('None') == 1
    assert vowels_count('negative') == 3
    assert vowels_count('boundary') == 3
    assert vowels_count('normal') == 2
    assert vowels_count('diverse') == 3
    assert vowels_count('100%') == 1

import pytest


def test_file1_method1():
    x = 5
    y = 6
    assert x+1 == y, "x+1 should equal y"


def test_file1_method2():
    x = 5
    y = 6
    assert x+1 == y, "test failed"

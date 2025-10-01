from lesson4 import hello

def test_default():
    assert hello() == "hello, world"

def test_arg():
    assert hello("David") == "hello, David"

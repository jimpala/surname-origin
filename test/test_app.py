import pytest

import app

def test_main():
    assert app.main() == 0
import utils
import os
from unittest import TestCase
from unittest import mock

class TestTextFileLoader:
    def test_gets_text_filepaths(self):
        """It can get the names of text files from a given directory."""
        loader = utils.TextFileLoader(os.path.join(os.getcwd(), 'data', 'dummy'))
        assert os.path.join(os.getcwd(), 'data', 'dummy', 'basic_names.txt') in loader.filepaths

    def test_can_transpose_unicode(self):
        """Check that unicodeToAscii static method works robustly."""
        assert utils.TextFileLoader.unicodeToAscii('František Kupka') == 'Frantisek Kupka'
        assert utils.TextFileLoader.unicodeToAscii('Božena Němcová') == 'Bozena Nemcova'

    def test_can_read_lines(self):
        """It can read names into list."""
        loader = utils.TextFileLoader(os.path.join(os.getcwd(), 'data', 'dummy'))
        loader.filepaths = [os.path.join(os.getcwd(), 'data', 'dummy', 'basic_names.txt')]
        assert utils.TextFileLoader.readLinesIntoList(loader.filepaths[0]) == [
            'James Pyne', 'George Pyne', 'Louise Pyne', 'Brian Butt'
        ]

    def test_can_create_dict(self):
        """It can create a dict of language -> names."""
        loader = utils.TextFileLoader('blank')
        loader.filepaths = [os.path.join(os.getcwd(), 'data', 'dummy', 'French.txt')]
        test_dict = loader.createDict()
        assert test_dict['French']
        assert len(test_dict['French']) == 277

class TestWordVectoriser(TestCase):
    def test_can_turn_letter_to_index(self):
        """It can return a consistent index to a given ASCII letter."""
        assert utils._letter_to_index('a') == 0
        assert utils._letter_to_index(';') == 55
        assert utils._letter_to_index('F') == 31

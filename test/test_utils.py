import torch

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

    def test_can_turn_letter_to_tensor(self):
        """It can return a <1 x n_letters> one-hot PyTorch tensor corresponding to a given letter."""
        tensor = utils._letter_to_tensor('c')
        assert list(tensor.size()) == [1, 57]

        dummy_tensor = torch.zeros(1, 57)
        dummy_tensor[0][2] = 1
        assert torch.equal(tensor, dummy_tensor)

    def text_can_turn_line_into_tensor(self):
        """It can return a <line_length x 1 x n_letters> tensor given a word string."""
        tensor = utils.word_to_tensor('abcd')
        assert list(tensor.size()) == [4, 1, 57]

        dummy_tensor = torch.zeros(4, 1, 57)
        for i in range(4):
            dummy_tensor[i][0][i] = 1
        assert torch.equal(tensor, dummy_tensor)
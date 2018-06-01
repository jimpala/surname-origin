import utils
import os

class TestUtils:
    def test_TextFileLoader_gets_names(self):
        """Check that the TestFileLoader gets the names from a given directory."""
        loader = utils.TextFileLoader(os.path.join(os.getcwd(), 'data', 'dummy'))
        assert loader.filepaths == [os.path.join(os.getcwd(), 'data', 'dummy', 'basic_names.txt')]

    def test_TextFileLoader_can_transpose_unicode(self):
        """Check that unicodeToAscii static method works robustly."""
        assert utils.TextFileLoader.unicodeToAscii('František Kupka') == 'Frantisek Kupka'
        assert utils.TextFileLoader.unicodeToAscii('Božena Němcová') == 'Bozena Nemcova'

    def test_TextFileLoader_can_read_lines(self):
        """Check that TextFileLoader can read names into list."""
        loader = utils.TextFileLoader(os.path.join(os.getcwd(), 'data', 'dummy'))
        assert utils.TextFileLoader.readLinesIntoList(loader.filepaths[0]) == [
            'James Pyne', 'George Pyne', 'Louise Pyne', 'Brian Butt'
        ]
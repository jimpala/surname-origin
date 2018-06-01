import glob
import os
import string
import unicodedata


class TextFileLoader:
    """This class finds name text files in a given directory, and can then process them into a dict.

    Args:
        text_files_dir (str): Filepath to directory with names text files situated within.

    Attributes:
        filepaths (:list:`str`): Filepaths of .txt files.
    """
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    def __init__(self, text_files_dir):
        self.filepaths = glob.glob(os.path.join(text_files_dir, '*.txt'))
        print(self.filepaths)

    @staticmethod
    def unicodeToAscii(unicode):
        return ''.join(
            c for c in unicodedata.normalize('NFD', unicode)
            if unicodedata.category(c) != 'Mn'
            and c in TextFileLoader.all_letters
        )

    @staticmethod
    def readLinesIntoList(filepath):
        lines = open(filepath, encoding='utf-8').read().strip().split('\n')
        return [TextFileLoader.unicodeToAscii(line) for line in lines]

    def createDict(self):
        names_dict = dict()
        for filename in self.filepaths:
            category = filename.split('/')[-1].split('.')[0]
            lines = TextFileLoader.readLinesIntoList(filename)
            names_dict[category] = lines
        return names_dict
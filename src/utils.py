import glob
import os
import string
import unicodedata

import torch


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

def _letter_to_index(letter):
    """This function takes a letter and returns an index corresponding to TextFileLoader.all_letters.

    Args:
        letter (str): Single character string, length one. Must be ASCII.

    Returns:
        int: Index corresponding to TextFileLoader.all_letters.

    Raises:
        ValueError: If `letter` is a string of other than length one.
        TypeError: If `letter` is not `str` type.
    """
    if type(letter) != str:
        raise TypeError('letter must be a string')
    if len(letter) != 1:
        raise ValueError('letter must be a string of length one')

    return TextFileLoader.all_letters.find(letter)

def _letter_to_tensor(letter):
    index = _letter_to_index(letter)

    tensor = torch.zeros(1, TextFileLoader.n_letters)
    tensor[0][index] = 1

    return tensor

def word_to_tensor(line):
    tensor = torch.zeros(len(line), 1, TextFileLoader.n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][_letter_to_index(letter)] = 1
    return tensor
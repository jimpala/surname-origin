import app
import torch
import utils
from unittest import TestCase
from unittest.mock import patch

class TestMain:
    def test_main_completes(self):
        """
        This test method checks that the main method runs to completion, returning exit code 0.
        """
        assert app.main() == 0

class TestRNN:
    def test_RNN_forward(self):
        """It contains a forward() method that takes a <1 x n_input> input tensor and <1 x n_hidden> tensor,
        concatenates, then processes them to return a <1 x n_output> output and a <1 x n_hidden> tensor for the
        next step.
        """
        input = 5
        hidden = 6
        output = 1
        nn = app.RNN(input, hidden, output)
        forward_result = nn.forward(torch.rand((1,input)), torch.rand((1,hidden)))
        assert [1, output] == list(forward_result[0].size())
        assert [1, hidden] == list(forward_result[1].size())

    def test_RNN_hidden_zeros(self):
        """It contains a hidden_zeros() method that returns a <1 x n_hidden> tensor of zeros."""
        input = 5
        hidden = 6
        output = 1
        nn = app.RNN(input, hidden, output)
        hidden_zeros_result = nn.hidden_zeros()
        assert torch.zeros(1, hidden).equal(hidden_zeros_result)

class TestModelHandler(TestCase):

    dummy_data = {'French': ['Jacques', 'Pierre'], 'English': ['Jack', 'Peter']}
    categories = tuple(dummy_data.keys())
    n_categories = len(categories)
    names = dummy_data['French'] + dummy_data['English']

    def test_ModelHandler_can_load_model_and_data(self):
        """It takes data dict stores an RNN and language->names data dict."""

        n_hidden = 100
        handler = app.ModelHandler(TestModelHandler.dummy_data, utils.TextFileLoader.all_letters, n_hidden)

        assert isinstance(handler.data, dict)
        assert handler.categories == tuple(TestModelHandler.dummy_data.keys())
        assert isinstance(handler.rnn, app.RNN)
        assert isinstance(handler.letters, str)

        assert handler.rnn.input_size == len(utils.TextFileLoader.all_letters)
        assert handler.rnn.hidden_size == n_hidden
        assert handler.rnn.output_size == len(handler.data.keys())

    def test_ModelHandler_can_determine_most_likely_category(self):
        """It has a helper function that can return the most likely category of an output, and its index."""
        tensor = torch.tensor([[1.1, 2.7]])

        handler = app.ModelHandler(TestModelHandler.dummy_data, utils.TextFileLoader.all_letters, 100)
        results = handler._most_likely_category(tensor)
        assert results[0] == 'English'
        assert results[1] == 1

    def test_ModelHandler_can_pick_random_training_sample(self):
        """It has a helper function that can return a random training sample. Return should be that training
        sample's category, name, input tensor representation and output representation.
        """
        handler = app.ModelHandler(TestModelHandler.dummy_data, utils.TextFileLoader.all_letters, 100)
        result = handler._random_training_sample()
        assert result[0] in TestModelHandler.categories
        assert result[1] in TestModelHandler.names
        assert tuple(result[2].size()) == (1, TestModelHandler.n_categories)
        assert tuple(result[3].size())[1] == 1 and tuple(result[3].size())[2] == len(handler.letters)

    def test_ModelHandler_train_iteration(self):
        """It has a training iteration method that can take an output and input (category and name) tensor,
        then use them to perform backprop, updating the RNN parameters with gradient descent, and then returning
        the output tensor and loss.
        """

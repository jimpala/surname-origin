import app
import torch
import utils

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

class TestModelHandler:
    def test_ModelHandler_can_load_model_and_data(self):
        """It takes data dict stores an RNN and language->names data dict."""
        data = {'French': ['Jacques', 'Pierre'], 'English': ['Jack', 'Peter']}
        n_hidden = 100
        handler = app.ModelHandler(data, utils.TextFileLoader.all_letters, n_hidden)

        assert isinstance(handler.data, dict)
        assert isinstance(handler.rnn, app.RNN)
        assert isinstance(handler.letters, str)

        assert handler.rnn.input_size == len(utils.TextFileLoader.all_letters)
        assert handler.rnn.hidden_size == n_hidden
        assert handler.rnn.output_size == len(handler.data.keys())


import random

import torch
import torch.nn as nn

import utils


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        output = self.log_softmax(output)
        return output, hidden

    def hidden_zeros(self):
        return torch.zeros(1, self.hidden_size)

class ModelHandler:
    def __init__(self, data, letters, n_hidden=128):
        self.data = data
        self.categories = tuple(data.keys())
        self.n_categories = len(self.categories)
        self.rnn = RNN(len(utils.TextFileLoader.all_letters), n_hidden, len(data.keys()))
        self.letters = letters

        self.criterion = nn.NLLLoss()

    def _most_likely_category(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.categories[category_i], category_i

    def _random_training_sample(self):
        category = self.categories[random.randint(0, len(self.categories) - 1)]
        name = self.data[category][random.randint(0, len(self.data[category]) - 1)]
        category_tensor = torch.zeros(1, self.n_categories)
        category_tensor[0][self.categories.index(category)] = 1
        name_tensor = utils.word_to_tensor(name)
        return category, name, category_tensor, name_tensor

    def _train_iteration(self, category_tensor, line_tensor, learning_rate=0.005):
        pass


def main():
    return 0

if __name__ == '__main__':
    main()
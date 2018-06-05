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
        # Category tensor is 1D, with the index of the true class.
        category_tensor = torch.tensor([self.categories.index(category)], dtype=torch.long)
        name_tensor = utils.word_to_tensor(name)
        return category, name, category_tensor, name_tensor

    def _train_iteration(self, learning_rate=0.005):
        hidden = self.rnn.hidden_zeros()
        self.rnn.zero_grad()

        # Randomly select a category and name.
        _, _, category_tensor, name_tensor = self._random_training_sample()

        # Forward pass.
        for i in range(name_tensor.size()[0]):
            output, hidden = self.rnn(name_tensor[i][:][:], hidden)

        # Backward pass.
        loss = self.criterion(output, category_tensor)
        loss.backward()

        # Update parameters.
        for p in self.rnn.parameters():
            p.data.add_(-learning_rate * p.grad)

        return output, loss.item()

    def train(self, n_iter=1, learning_rate=0.05, output_losses=True):
        if output_losses:
            losses = list()
        for i in range(n_iter):
            _, iter_loss = self._train_iteration(learning_rate=learning_rate)
            if i % 1000 == 0:
                print("Completed training iteration {:d}".format(i))
            if output_losses:
                losses.append(iter_loss)
        if output_losses:
            return list(enumerate(losses, start=1))

    def _evaluate(self, name_tensor):
        hidden = self.rnn.hidden_zeros()

        for i in range(name_tensor.size()[0]):
            output, hidden = self.rnn(name_tensor[i][:][:], hidden)

        return output

    def predict(self, name, top_predictions=1):
        with torch.no_grad():
            output = self._evaluate(utils.word_to_tensor(name))

            # Get top N categories
            topv, topi = output.topk(top_predictions, 1, True)
            predictions = []

            for i in range(top_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, self.categories[category_index]))
                predictions.append([value, self.categories[category_index]])

        return predictions


def main():
    return 0

if __name__ == '__main__':
    main()
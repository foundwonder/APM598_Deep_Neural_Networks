import numpy as np
import pandas as pd
from math import floor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class TwoLayerReluModule(nn.Module):
    def __init__(self, in_feature: int, hidden_neuron: int, out_feature: int):
        super(TwoLayerReluModule, self).__init__()
        self._model = nn.Sequential(nn.Linear(in_feature, hidden_neuron),
                                    nn.ReLU(),
                                    nn.Linear(hidden_neuron, out_feature))

    def forward(self, x):
        z = self._model(x)
        return z


class TwoLayerLinearClassificationNN:
    r"""
    This class is a two layer nn with Relu
    """
    def __init__(self, data_x, data_y, hidden_neurons=2,
                 learning_rate=0.05, momentum=0.9,
                 batchsize=5, optim='SDG'):
        self._data_x, self._data_y = data_x, data_y
        self._tensor_x = torch.as_tensor(self._data_x, dtype=float)
        self._tensor_y = torch.as_tensor(self._data_y, dtype=torch.long)
        self._num_classes = torch.unique(self._tensor_y).size()[0]  # how many classes
        self._N, self._nX = self._tensor_x.size()  # _N is the number of training data, _nX is the number of x params.
        self._input_features = self._nX
        self._model = TwoLayerReluModule(self._input_features, hidden_neurons, self._num_classes)
        self._loss_function = nn.CrossEntropyLoss()
        # self._loss_function = nn.MSELoss()
        self._learning_rate, self._momentum, self._batchsize, self._optim = learning_rate, momentum, batchsize, optim
        self._training_dataset = TensorDataset(self._tensor_x, self._tensor_y)
        self._loader_train = DataLoader(self._training_dataset, batch_size=self._batchsize, shuffle=True)
        self._nbr_minibatch_train = len(self._loader_train)
        self._df = pd.DataFrame(columns=('epoch', 'loss_train', 'accuracy_train', 'parameters'))

    @property
    def model(self) -> nn.Module:
        return self._model

    def print_parameters(self):
        for param in self._model.named_parameters():
            print(param)

    @staticmethod
    def find_y_class(y):
        # y is a list/tensor
        scores_array = np.asarray(y.tolist())
        return np.argmax(scores_array) + 1

    def train_model(self, num_epoch):
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=self._momentum)
        if self._optim == 'Adam':
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

        for epoch in range(num_epoch):

            step_to_print = floor(num_epoch/10)
            running_loss_train, accuracy_train = 0.0, 0.0
            self._model.train()
            for X, y_train in self._loader_train:
                optimizer.zero_grad()
                y = y_train - 1
                # one-hot coding, no need
                # y_train = y_train - 1
                # y = torch.zeros(self._batchsize, self._num_classes)
                # y[range(y.shape[0]), y_train] = 1
                score = self._model(X.float())
                loss = self._loss_function(score, y)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.detach().numpy()
                accuracy_train += (score.argmax(dim=1) == y).sum().numpy()
            self._model.eval()
            parameters = [param for name, param in self._model.named_parameters()]
            loss_train = running_loss_train/self._nbr_minibatch_train
            accuracy_train /= self._N
            if epoch%step_to_print == 0:
                print(f'-- epoch {epoch}')
                print(f'loss is {loss_train}, accuracy is {accuracy_train}')
            self._df.loc[epoch] = [epoch, loss_train, accuracy_train, parameters]
        print(f'{num_epoch} trainings is finished!')
        print(f'Final loss is {loss_train}, accuracy is {accuracy_train}')

    def find_accurate_params_after_training(self, threshold=0.9, nbr_of_good_params=4):
        accurate_df = self._df.loc[self._df.accuracy_train >= threshold]
        if len(accurate_df) == 0:
            print(f'There is no models accuracy larger than {threshold}')
        else:
            print(f'\n The parameters that perform with (at least) {threshold*100}% accuracy are:')
            number_to_print = min(len(accurate_df), nbr_of_good_params)
            for i in range(number_to_print):
                df = accurate_df.iloc[i]
                epoch, loss, accuracy, parameters = df['epoch'], df['loss_train'], df['accuracy_train'], df['parameters']
                print(f'\n{epoch} trainings, accuracy is {accuracy}, loss is {loss}')
                for name, params in parameters:
                    print(params)

    def plot_loss_accuracy(self):
        r"""
        Plot the loss and predicting accuracy during the training. x=epoch, y=loss or accuracy.
        :param num_epoch: number of epochs
        :param learning_rate: by defaut is 0.05
        :return: does not return anything, only plot two graphs for loss and accuracy.
        """
        df = self._df
        plt.figure(1)
        plt.grid(True)
        plt.clf()
        # plt.plot(df['epoch'], df['accuracy_train'], marker='o', color='red')
        plt.plot(df['epoch'], df['accuracy_train'], color='red')
        # plt.plot(df['epoch'], df['accuracy_test'], marker='o', color='orange', linestyle='dashed')
        # plt.plot(df['epoch'], df['loss_train'], marker='o', color='blue')
        plt.plot(df['epoch'], df['loss_train'], color='blue')
        # plt.plot(df['epoch'], df['loss_test'], marker='o', color='teal', linestyle='dashed')

        plt.xlabel(r'epoch')
        plt.ylabel(r'loss/accuracy')
        plt.legend(['accuracy_train', 'loss_train'])
        plt.show()

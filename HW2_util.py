"""
author: Jieshu Wang (jwang490@asu.edu, foundwonder@gmail.com)
for APM598: Intro to Deep Neural Networks at ASU, Spring 2020.
this is a util file containing:
1) a dataclass that holds hyper-parameters for neural networks and trainers;
2) several customized neural networks;
3) a trainer class that takes a data class and a neural network and train it with dropout and minibatches;
4) some functions for plot and print the training result.
"""

import pandas as pd
from math import floor
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from dataclasses import dataclass


@dataclass
class ModuleParams:
    r"""
    This dataclass holds the hyper-parameters needed for the modules.
    """
    num_classes: int = 10
    middle_channel: int = 4
    drop_out_rate: float = 0.2
    num_epoch: int = 5
    batch_size: int = 20
    learning_rate: float = 0.0001
    momentum: float = 0.9
    print_number: int = 5
    optimizer: str = 'Adam'
    linear_in_features: int = 2
    linear_hidden_features: int = 2
    linear_out_features: int = 2
    num_good_parameters: int = 5
    good_parameter_threshold: float = 0.95
    loss_function: str = 'cross_entropy'


class TwoLayerReluClassificationModel(nn.Module):
    r"""
    This class is a two-layer linear neural network with a ReLu activation function on the first layer.
    Basically, f(x) = b2 + w2 * σ(b1 + w1 * x)
    """
    def __init__(self, model_params: ModuleParams):
        self._in_feature = model_params.linear_in_features
        self._hidden_neurons = model_params.linear_hidden_features
        self._out_feature = model_params.linear_out_features
        super(TwoLayerReluClassificationModel, self).__init__()
        self._model = nn.Sequential(nn.Linear(self._in_feature, self._hidden_neurons),
                                    nn.ReLU(),
                                    nn.Linear(self._hidden_neurons, self._out_feature))

    def forward(self, x):
        z = self._model(x)
        return z


class CNNNet(nn.Module):
    r"""
    This class is a CNN that orderly consists of one Conv net, one ReLu activation function,
    one max pool, a second Conv net, a second ReLu, and a linear classifier.
    Drop out rate by default is set as 0.2.
    The dataset fitting in this CNN is MNIST, specifically, Fashion-MNIST (dimension 28*28),
    that's why the flattern layer dimension is 9*9 (Conv: 28-(3-1)=26, maxpool: 26/2=13, Conv:13-(5-1)=9).
    If image size changes, can add dimensions to ModelParams, and pass it here.
    """
    def __init__(self, model_params: ModuleParams):
        super(CNNNet, self).__init__()
        self._middle_channel = model_params.middle_channel
        self._features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self._middle_channel, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=self._middle_channel, out_channels=self._middle_channel * 2, kernel_size=5),
            nn.ReLU()
        )
        self._classifier = nn.Sequential(
            nn.Dropout(p=model_params.drop_out_rate),
            nn.Linear(self._middle_channel * 2 * 9 * 9, model_params.num_classes)
        )

    def forward(self, x):
        x = self._features(x)
        x = torch.flatten(x, 1)
        s = self._classifier(x)
        return s


class MyNNTrainer:
    r"""
    a generic NN trainer
    """
    def __init__(self, model: nn.Module, model_params: ModuleParams,
                 training_set, test_set=None):
        self._training_set, self._test_set = training_set, test_set
        self._model = model(model_params=model_params)
        self._loss_function = nn.CrossEntropyLoss()
        if model_params.loss_function == 'MSE':
            self._loss_function = nn.MSELoss()
        self._learning_rate, self._num_epoch = model_params.learning_rate, model_params.num_epoch
        self._batchsize, self._optimizer = model_params.batch_size, model_params.optimizer
        self._momentum, self._print_num = model_params.momentum, model_params.print_number
        self._num_good_parameters, self._threshold = model_params.num_good_parameters, model_params.good_parameter_threshold
        self._loader_train = DataLoader(self._training_set, batch_size=self._batchsize, shuffle=True)
        self._N_training_data = len(self._training_set)
        self._nbr_minibatch_train = len(self._loader_train)
        self._df = pd.DataFrame(columns=('epoch', 'loss_train', 'accuracy_train', 'parameters'))
        if self._test_set is not None:
            self._loader_test = DataLoader(self._test_set, batch_size=self._batchsize, shuffle=False)
            self._N_test_data = len(self._test_set)
            self._nbr_minibatch_test = len(self._loader_test)
            self._df = pd.DataFrame(columns=('epoch', 'loss_train', 'loss_test', 'accuracy_train', 'accuracy_test','parameters'))
        self.train()

    def train(self):
        # if self._optimizer == 'Adam':
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        if self._optimizer == 'SGD':
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=self._momentum)
        t0 = time.time()
        good_parameters_counter = 0
        for epoch in range(self._num_epoch):
            parameters = 0
            step_to_print = floor(self._num_epoch / self._print_num)
            # train
            running_loss_train, accuracy_train = 0.0, 0.0
            self._model.train()
            for X, y in self._loader_train:
                # 1) initialize the gradient "loss" to zero
                optimizer.zero_grad()
                # 2) compute the score and loss
                score = self._model(X.float())
                # score = self._model(X)
                loss = self._loss_function(score, y)
                # 3) estimate the gradient and update parameters
                loss.backward()
                optimizer.step()
                # 4) estimate the overall loss over the all training set
                running_loss_train += loss.detach().numpy()
                accuracy_train += (score.argmax(dim=1) == y).sum().numpy()
            # test
            if self._test_set is not None:
                running_loss_test, accuracy_test = 0.0, 0.0
                self._model.eval()
                with torch.no_grad():
                    for X, y in self._loader_test:
                        # 1) computer the score and loss
                        # score = self._model(X)
                        score = self._model(X.float())
                        loss = self._loss_function(score, y)
                        # 2) estimate the overall loss over the all test set
                        running_loss_test += loss.detach().numpy()
                        accuracy_test += (score.argmax(dim=1) == y).sum().numpy()
            # end epoch and statistics
            loss_train = running_loss_train / self._nbr_minibatch_train
            accuracy_train /= self._N_training_data
            if accuracy_train >= self._threshold and good_parameters_counter < self._num_good_parameters:
                parameters = [(name, param) for name, param in self._model.named_parameters()]
                good_parameters_counter += 1
            if self._test_set is not None:
                loss_test = running_loss_test / self._nbr_minibatch_test
                accuracy_test /= self._N_test_data
            if epoch % step_to_print == 0:
                print(f'-- epoch {epoch} --')
                if self._test_set is not None:
                    print(f'    loss (train, test): {loss_train:.4f}, {loss_test:.4f}')
                    print(f'    accuracy (train, test): {accuracy_train:.4f}, {accuracy_test:.4f}')
                else:
                    print(f'    loss: {loss_train:.4f}')
                    print(f'    accuracy: {accuracy_train:.4f}')
            if self._test_set is not None:
                self._df.loc[epoch] = [epoch, loss_train, loss_test, accuracy_train, accuracy_test, parameters]
            else:
                self._df.loc[epoch] = [epoch, loss_train, accuracy_train, parameters]

        t1 = time.time()
        print(f'{self._num_epoch} trainings is finished! Spent time {t1 - t0} seconds.')
        if self._test_set is not None:
            print(f'Final loss (train, test): {loss_train:.4f}, {loss_test:.4f}')
            print(f'Final accuracy (train, test): {accuracy_train:.4f}, {accuracy_test:.4f}')
        else:
            print(f'Final training loss: {loss_train:.4f}')
            print(f'Final training accuracy: {accuracy_train:.4f}')

    @property
    def result(self) -> pd.DataFrame:
        return self._df

    @property
    def model(self):
        return self._model

def plot_loss_accuracy(df: pd.DataFrame, marker='None', include_test=True):
    r"""
    :param df: DataFrame from the MyNNTrainer.result
    :param marker: usually 'o', 'D', '^', 'v', '<', '>', 's', 'p', '*'.
    documentation can be found at https://matplotlib.org/3.1.1/api/markers_api.html
    :param include_test:
    :return:
    """
    plt.figure(1)
    plt.grid(True)
    plt.clf()
    plt.plot(df['epoch'], df['accuracy_train'], marker=marker, color='red')
    if include_test is True:
        plt.plot(df['epoch'], df['accuracy_test'], color='orange', marker=marker, linestyle='dashed')
    plt.plot(df['epoch'], df['loss_train'], marker=marker, color='blue')
    if include_test is True:
        plt.plot(df['epoch'], df['loss_test'], color='teal', marker=marker, linestyle='dashed')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss/accuracy')
    if include_test is True:
        plt.legend(['accuracy train', 'accuracy test', 'loss_train', 'loss_test'])
    else:
        plt.legend(['accuracy train', 'loss_train'])
    plt.show()


def print_good_parameters(df: pd.DataFrame, dataset: ModuleParams):
    threshold = dataset.good_parameter_threshold
    accurate_df = df.loc[df.parameters != 0]
    if len(accurate_df) == 0:
        print(f'There is no models accuracy larger than {threshold}')
    else:
        print(f'\n The parameters that perform with (at least) {threshold * 100}% accuracy are:')
        for index, row in accurate_df.iterrows():
            epoch, loss, accuracy, parameters = row['epoch'], row['loss_train'], row['accuracy_train'], row['parameters']
            print(f'\n-- epoch {epoch}: (accuracy, loss): ({accuracy:.4f}, {loss:.4f})')
            for name, params in parameters:
                print(f'    {name}')
                print(f'    {params}')


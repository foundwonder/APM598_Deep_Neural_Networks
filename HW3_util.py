import random
import pandas as pd
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from dataclasses import dataclass
from HW2_util import MyNNTrainer
import string
from math import log10 as log
from numpy.linalg import norm
import numpy as np


all_letters = string.ascii_letters + string.digits + " '.,;:!?-()\"" + '\n'
num_of_letters = len(all_letters)


def letter_to_index(letter, all_letter: str):
    r""" Find letter index from all_letters, e.g. "a" = 0
    :param letter: str
    :return: the index of the letter
    """
    return all_letter.find(letter)


def letter_to_tensor(letter, all_letter: str):
    r""" Transform a letter into a 'hot-vector' (tensor)
    :param letter:
    :return:
    """
    num_of_letter = len(all_letter)
    letter_tensor = torch.zeros(1, num_of_letter)
    letter_tensor[0][letter_to_index(letter, all_letter)] = 1
    return letter_tensor


@dataclass
class ModuleParamsVanillaRNN:
    all_letters_base: str = all_letters
    hidden_size: int = 2
    learning_rate: float = 0.005
    n_steps: int = 1000
    chunk_size: int = 30
    optimizer_name: str = 'Adam'
    is_print_training: bool = False
    printing_step: int = 50


class VanillaRNN(nn.Module):
    r""" This class is from notebook4_n_gram. credit: Prof. Sebastien Motsch @ ASU
    The vanilla RNN: from (x_t,h_t-1) input,hidden-state
    h_t = tanh( R*h_t-1 + A*x_t)
    y_t = B*h_t
    where A is the encoder, B the decoder, R the recurrent matrix
    """
    def __init__(self, module_params: ModuleParamsVanillaRNN):
        super(VanillaRNN, self).__init__()
        self._in_out_size = len(module_params.all_letters_base)
        self._hidden_size = module_params.hidden_size
        self._A = nn.Linear(self._in_out_size, self._hidden_size, bias=False)
        self._R = nn.Linear(self._hidden_size, self._hidden_size, bias=False)
        self._B = nn.Linear(self._hidden_size, self._in_out_size, bias=False)
        self._tanh = nn.Tanh()

    def forward(self, x, h):
        # update the hidden state
        h_update = self._tanh(self._R(h) + self._A(x))
        # prediction
        y = self._B(h_update)
        # y = F.softmax(y, dim=1)
        return y, h_update

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size)


class VanillaRNNTrainer:
    def __init__(self, module_params: ModuleParamsVanillaRNN, module: nn.Module, training_x, training_y):
        self._learning_rate = module_params.learning_rate
        self._chunk_size = module_params.chunk_size
        self._n_steps = module_params.n_steps
        self._module = module(module_params=module_params)
        self._optimizer_name = module_params.optimizer_name
        self._loss_function = nn.CrossEntropyLoss()
        self._df = pd.DataFrame(columns=('step', 'loss'))
        self._training_x = training_x
        self._training_y = training_y
        self._is_print_training = module_params.is_print_training
        self._printing_step = module_params.printing_step
        self._all_letters = module_params.all_letters_base
        self.train()

    def train(self):
        optimizer = torch.optim.Adam(self._module.parameters(), lr=self._learning_rate)
        for step in range(self._n_steps):
            h = self._module.init_hidden()
            optimizer.zero_grad()
            loss = 0.0
            start_index = random.randint(0, len(self._training_x) - self._chunk_size)
            end_index = start_index + self._chunk_size + 1
            chunk_x = self._training_x[start_index:end_index]
            chunk_y = self._training_y[start_index:end_index]
            if self._is_print_training and step % self._printing_step == 0:
                print(f'  input = {chunk_x}')
                chunk_predicted = '' # chunk_y[0]
            for p in range(self._chunk_size):
                x = letter_to_tensor(chunk_x[p], self._all_letters)
                letter_x_next = letter_to_index(chunk_y[p], self._all_letters)
                target = torch.tensor([letter_x_next], dtype=torch.long)
                y, h = self._module(x, h)
                y = F.softmax(y, dim=1)
                loss += self._loss_function(y.view(1, -1), target)
                if self._is_print_training:
                    chunk_predicted += self._all_letters[y.argmax()]
            loss.backward()
            optimizer.step()
            ave_loss = loss.detach().numpy() / self._chunk_size
            if self._is_print_training and step%self._printing_step == 0:
                print(f'  output = {chunk_predicted}')
            self._df.loc[step] = [step, ave_loss]
            if step%self._printing_step == 0:
                print(f'loss at step {step}: {ave_loss}')

    @property
    def result(self) -> pd.DataFrame:
        return self._df

    @property
    def model(self):
        return self._module


def forward_certain_times(num_times: int, x_1, x_n, h_0, module, module_params):
    initial_module = module(module_params)
    y, h = initial_module(x_1, h_0)
    for i in range(num_times):
        y, h = initial_module(x_n, h)
    return y


def compute_y_difference(num_times: int, x_1, x_n, h_0, perturbating_func,
                         perturbation_values, module, module_params, precision=4):
    torch.set_printoptions(precision=precision)  # because sometimes the difference is too small
    df = pd.DataFrame(columns=['perturbation', 'perturbation_log',
                               'y_difference', 'y_difference_log'])
    for perturbation_value in perturbation_values:
        perturbation_log = log(perturbation_value)
        y = forward_certain_times(num_times, x_1, x_n, h_0, module, module_params)
        x_1_perturbated = perturbating_func(perturbation_value)
        y_perturbated = forward_certain_times(num_times, x_1_perturbated, x_n, h_0, module, module_params)
        difference_y = (y-y_perturbated).detach().numpy()
        difference_y_norm = norm(np.asarray(difference_y))
        try:
            difference_y_log = log(difference_y_norm)
        except ValueError:
            difference_y_log = '-∞'
        df.loc[perturbation_value] = [perturbation_value, perturbation_log,
                                      difference_y_norm, difference_y_log]
    return df





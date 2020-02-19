import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TwoLayerLinearClassificationNN:
    r"""
    This class is a two layer nn with Relu
    """
    def __init__(self, data_x, data_y, learning_rate=0.05, momentum=0.9):
        r"""
        :param data_x:
        :param data_y: a list / array of class number starting from 1, e.g. 2 instead of a array or tensor, e.g. [0,1]
        """
        self._data_x = data_x
        self._class_y = data_y
        self._tensor_x = torch.as_tensor(self._data_x, dtype=float)
        self._tensor_class_y = torch.as_tensor(self._class_y)
        self._num_classes = torch.unique(self._tensor_class_y).size()[0]  # how many classes
        self._N, self._nX = self._tensor_x.size()  # _N is the number of training data, _nX is the number of x params.

        # This for loop is to transfer the label of class into a tensor
        # e.g. if data_y = [1,2,2], then it will be transfer into a tensor([1,0],[0,1],[0,1])
        self._data_y = []
        for i in self._class_y:

            class_y = i - 1
            y_dummy = []
            for j in range(self._num_classes):
                if j == class_y:
                    y_dummy.append(1)
                else:
                    y_dummy.append(0)
            self._data_y.append(y_dummy)
        self._tensor_y = torch.as_tensor(self._data_y)

        self._model = nn.Sequential(nn.Linear(2, 2),
                                    nn.ReLU(),
                                    nn.Linear(2, self._num_classes))
        self._loss_function = nn.MSELoss(reduction='mean')
        # self._loss_function = nn.CrossEntropyLoss()
        self._learning_rate = learning_rate
        self._momentum = momentum

    @property
    def model(self) -> nn.Module:
        return self._model

    def forward(self):
        y_hat = self._model(self._tensor_x.float().view(self._N, self._nX))
        return y_hat

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
        accuracy_list = []
        losses = []
        for epoch in range(num_epoch):
            running_loss = 0.0
            optimizer.zero_grad()
            tensor_y_hat = self.forward()
            loss = self._loss_function(tensor_y_hat, self._tensor_y.float().view(self._N, self._num_classes))
            loss.backward()
            running_loss += loss.detach().numpy()
            optimizer.step()
            accuracy = self.measure_accuracy_in_training_data()
            accuracy_list.append(accuracy)
            losses.append(loss)
        return loss, accuracy_list, losses

    def train_model_and_print(self, num_epoch):
        r"""
        this method is just for train and print the final loss, not much use.
        :param num_epoch:
        :param learning_rate:
        :return:
        """
        loss, _, _ = self.train_model(num_epoch)
        for param in self._model.named_parameters():
            print(param)
        print(f'The loss is {loss}')

    def measure_accuracy_in_training_data(self):
        r"""
        This method does not train the model, only returns the accuracy of self._model
        :return: the percentage of the accurate predictions in the training data
        """
        y_hat = self._model(self._tensor_x.float().view(self._N, self._nX))
        y_hat_list = y_hat.tolist()
        y_hat_classes = []
        for i in y_hat_list:
            score_array = np.asarray(i)
            class_of_y = np.argmax(score_array) + 1
            y_hat_classes.append(class_of_y)

        # compare y_hat_classes with data_y
        true_counter = 0
        for i in range(self._N):
            if y_hat_classes[i] == self._class_y[i]:
                true_counter += 1
        percentage = true_counter/self._N
        return percentage

    def find_good_parameters(self, num_epoch: int, threshold=0.9, num_of_good_models=4):
        r"""
        I have a difficult time finding the w and b that make #1a predict 100%, so I wrote this method.
        Not sure if this is correct.

        :param num_epoch: number of epochs of training. Each epoch train on the entire data.
        :param learning_rate: by default 0.05. Maybe can set another number?
        :param threshold:
        :param num_of_good_models:
        :return: It does not return anything, only prints out the parameters of the good models during training.
        """
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=self._momentum)
        counter = 0
        for epoch in range(num_epoch):
            running_loss = 0.0
            optimizer.zero_grad()
            tensor_y_hat = self.forward()
            loss = self._loss_function(tensor_y_hat, self._tensor_y.float().view(self._N, self._num_classes))
            loss.backward()
            running_loss += loss.detach().numpy()
            optimizer.step()
            accuracy = self.measure_accuracy_in_training_data()
            if accuracy > threshold:
                print(f'\n{epoch} trainings, model accuracy is {accuracy}')
                for param in self._model.named_parameters():
                    print(param)
                print(f'The loss is {loss}')
                counter += 1
            if counter == num_of_good_models:
                break
        if counter == 0:
            print(f'There is no models accuracy larger than {threshold}')

    def plot_loss_accuracy(self, num_epoch, learning_rate=0.05):
        r"""
        Plot the loss and predicting accuracy during the training. x=epoch, y=loss or accuracy.
        :param num_epoch: number of epochs
        :param learning_rate: by defaut is 0.05
        :return: does not return anything, only plot two graphs for loss and accuracy.
        """
        _, accuracies, losses = self.train_model(num_epoch)
        plot_x = range(num_epoch)
        plt.grid(True)
        plt.plot(plot_x, accuracies, '-', c='r')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title("accuracy of the model")
        plt.show()

        plt.grid(True)
        plt.plot(plot_x, losses, '-', c='b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("loss of the model")
        plt.show()

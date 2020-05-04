# In this file, the code for Generator and Discriminator are from https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x/blob/master/Chapter04/dcgan.py
# The code for DCGANTrainer is modified based on https://github.com/PacktPublishing/Hands-On-Generative-Adversarial-Networks-with-PyTorch-1.x/blob/master/Chapter04/dcgan.py
# The code is from book: John Hany and Greg Walters. 2019. Hands-On Generative Adversarial Networks with PyTorch 1.x: Implement next-generation neural networks to build powerful GAN models using Python. Packt Publishing Ltd.

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from fid_score import calculate_fid_given_paths
import utils


@dataclass
class GANParams:
    cuda: bool = True  # Change to False for CPU training
    data_path: str = '../data_sets/MNIST/MNIST'
    output_path: str = './output_20200430/output_MNIST'
    batch_size: int = 32  # 128, Adjust this value according to your GPU memory
    image_channel: int = 1
    z_dim: int = 100
    g_hidden: int = 64
    x_dim: int = 64
    d_hidden: int = 64
    num_epoch: int = 25
    real_label: int = 1
    fake_label: int = 0
    learning_rate: float = 2e-4
    seed: int = 1
    dataset_name: str = 'MNIST'
    lsun_class = ['bedroom_train']
    loss_function_name: str = "BCELoss"
    printing_step: int = 100
    plotting_steps: int = 100


MNIST_params = GANParams(batch_size=64)
galaxy_params = GANParams(data_path='../galaxy-zoo-the-galaxy-challenge',
                          output_path='./output_20200430/output_galaxy_zoo',
                          image_channel=3, batch_size=64)
celeba_params = GANParams(data_path='../data_sets/celebA/',
                          output_path='./output_20200430/output_celebA',
                          image_channel=3, batch_size=64)


class Generator(nn.Module):
    def __init__(self, params: type(GANParams)):
        super(Generator, self).__init__()
        self._image_channel = params.image_channel
        self._g_hidden = params.g_hidden
        self._z_dim = params.z_dim
        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(self._z_dim, self._g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self._g_hidden * 8),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(self._g_hidden * 8, self._g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._g_hidden * 4),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(self._g_hidden * 4, self._g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._g_hidden * 2),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(self._g_hidden * 2, self._g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._g_hidden),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(self._g_hidden, self._image_channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, params: type(GANParams)):
        super(Discriminator, self).__init__()
        self._image_channel = params.image_channel
        self._d_hidden = params.d_hidden

        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(self._image_channel, self._d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(self._d_hidden, self._d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(self._d_hidden * 2, self._d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(self._d_hidden * 4, self._d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self._d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(self._d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class DCGANTrainer:
    def __init__(self, params: GANParams, generator: type(nn.Module), discriminator: type(nn.Module)):
        self._cuda = params.cuda and torch.cuda.is_available()
        self._data_path = params.data_path
        self._output_path = params.output_path
        self._batch_size = params.batch_size
        self._image_channel = params.image_channel
        self._z_dim = params.z_dim
        self._g_hidden = params.g_hidden
        self._x_dim = params.x_dim
        self._d_hidden = params.d_hidden
        self._num_epoch = params.num_epoch
        self._real_label = params.real_label
        self._fake_label = params.fake_label
        self._learning_rate = params.learning_rate
        self._seed = params.seed
        self._dataset_name = params.dataset_name
        self._lsun_class = params.lsun_class
        self._loss_function_name = params.loss_function_name
        self._generator = generator
        self._discriminator = discriminator
        self._log_file = os.path.join(self._output_path, 'log.txt')
        self._printing_step = params.printing_step
        self._criterion = nn.BCELoss()
        self._result_df = pd.DataFrame(columns=('epoch', 'num', 'loss_d_real', 'loss_d_fake', 'loss_g'))
        self._plotting_step = params.plotting_steps

    def __loss_function(self):
        self._criterion = None
        if self._loss_function_name == 'BCELoss':
            self._criterion = nn.BCELoss()
        elif self._loss_function_name == 'CrossEntropyLoss':
            self._criterion = nn.CrossEntropyLoss()
        elif self._loss_function_name == 'MSELoss':
            self._criterion = nn.MSELoss

    def __dataset(self, dataset_name):
        r"""
        initialize dataset based on the name of the dataset
        :return: dataset
        """
        if dataset_name == 'MNIST':
            dataset = dset.MNIST(root=self._data_path, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(self._x_dim),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))
                                 ]))
        elif dataset_name == 'FashionMNIST':
            dataset = dset.FashionMNIST(root=self._data_path, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(self._x_dim),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                        ]))
        elif dataset_name == 'lsun':
            dataset = dset.LSUN(root=self._data_path, classes=self._lsun_class,
                                transform=transforms.Compose([
                                    transforms.Resize(self._x_dim),
                                    transforms.CenterCrop(self._x_dim),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        else:
            dataset = dset.ImageFolder(root=self._data_path,
                                       transform=transforms.Compose([
                                           transforms.Resize(self._x_dim),
                                           transforms.CenterCrop(self._x_dim),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        return dataset

    def _clear_folder(self):
        utils.clear_folder(self._output_path)
        print(f"Logging to {self._log_file}\n")
        sys.stdout = utils.StdOut(self._log_file)
        print(f"PyTorch version: {torch.__version__}")
        if self._cuda:
            print(f"CUDA version: {torch.version.cuda}\n")

    def _create_seed(self):
        if self._seed is None:
            self._seed = np.random.randint(1, 10000)
        print("Random Seed: ", self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if self._cuda:
            torch.cuda.manual_seed(self._seed)
        cudnn.benchmark = True  # May train faster but cost more memory

    @staticmethod
    def _weights_init(m):
        r"""
        custom weights initialization
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _set_up_nets(self):
        self._dataset = self.__dataset(self._dataset_name)
        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=self._batch_size,
                                                        shuffle=True, num_workers=4)
        self._device = torch.device("cuda:0" if self._cuda else "cpu")
        assert self._dataset

        self._net_g = self._generator.to(self._device)
        self._net_g.apply(self._weights_init)
        print(self._net_g)

        self._net_d = self._discriminator.to(self._device)
        self._net_d.apply(self._weights_init)
        print(self._net_d)

        self.__loss_function()

        self._viz_noise = torch.randn(self._batch_size, self._z_dim, 1, 1, device=self._device)

        self._optimizer_d = optim.Adam(self._net_d.parameters(), lr=self._learning_rate, betas=(0.5, 0.999))
        self._optimizer_g = optim.Adam(self._net_g.parameters(), lr=self._learning_rate, betas=(0.5, 0.999))

    @property
    def result(self) -> pd.DataFrame:
        return self._result_df

    def plot_result(self, is_plot_loss_d_real: bool = True, is_plot_loss_d_fake: bool = True,
                    is_plot_loss_g: bool = False):
        plt.clf()
        if is_plot_loss_d_real is True:
            plt.plot(self._result_df['epoch'], self._result_df['loss_d_real'], label='loss_d_real')
        if is_plot_loss_d_fake is True:
            plt.plot(self._result_df['epoch'], self._result_df['loss_d_fake'], label='loss_d_fake')
        if is_plot_loss_g is True:
            plt.plot(self._result_df['epoch'], self._result_df['loss_g'], label='loss_g')
        plt.grid()
        plt.xlabel('epoch')
        plt.legend()
        plt.ylabel('loss')
        plt.show()

    # @TODO:
    def fid_score(self):
        return calculate_fid_given_paths([self._data_path, self._output_path], 32, self._cuda, 2048)

    def train(self):
        self._clear_folder()
        self._create_seed()
        self._set_up_nets()
        start_time = time.time()

        for epoch in range(self._num_epoch):
            for i, data in enumerate(self._data_loader):
                x_real = data[0].to(self._device)
                real_label = torch.full((x_real.size(0),), self._real_label, device=self._device)
                fake_label = torch.full((x_real.size(0),), self._fake_label, device=self._device)

                # Update D with real data
                self._net_d.zero_grad()
                y_real = self._net_d(x_real)
                loss_d_real = self._criterion(y_real, real_label)
                loss_d_real.backward()

                # Update discriminator with fake data
                z_noise = torch.randn(x_real.size(0), self._z_dim, 1, 1, device=self._device)
                x_fake = self._net_g(z_noise)
                y_fake = self._net_d(x_fake.detach())
                loss_d_fake = self._criterion(y_fake, fake_label)
                loss_d_fake.backward()
                self._optimizer_d.step()

                # Update generator with fake data
                self._net_g.zero_grad()
                y_fake_r = self._net_d(x_fake)
                loss_g = self._criterion(y_fake_r, real_label)
                loss_g.backward()
                self._optimizer_g.step()

                if i % self._plotting_step == 0:
                    self._result_df.loc[epoch] = [epoch, i, loss_d_real.mean().item(), loss_d_fake.mean().item(),
                                                  loss_g.mean().item()]

                if i % self._printing_step == 0:
                    print(
                        f'{(time.time() - start_time) / 60:.1f} mins: Epoch {epoch} [{i}/{len(self._data_loader)}] loss_d_real: {loss_d_real.mean().item():.4f} loss_d_fake: {loss_d_fake.mean().item():.4f} loss_G: {loss_g.mean().item():.4f}')
                    vutils.save_image(x_real, os.path.join(self._output_path, 'real_samples.png'), normalize=True)
                    with torch.no_grad():
                        viz_sample = self._net_g(self._viz_noise)
                        vutils.save_image(viz_sample, os.path.join(self._output_path, f'fake_samples_{epoch}.png'),
                                          normalize=True)
            torch.save(self._net_g.state_dict(), os.path.join(self._output_path, f'net_g_{epoch}.pth'))
            torch.save(self._net_d.state_dict(), os.path.join(self._output_path, f'net_d_{epoch}.pth'))


def set_and_run_dcgan(params: type(GANParams)):
    r"""
    this method passes parameters to generator, discriminator, and DCGAN trainer and run
    :param params:
    :return:
    """
    generator = Generator(params)
    discriminator = Discriminator(params)
    dcgan_trainer = DCGANTrainer(params, generator, discriminator)
    dcgan_trainer.train()
    return dcgan_trainer


def fid_score(output_path, batch_size, cuda, dims):
    pass
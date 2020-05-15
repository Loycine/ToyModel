import random
import math

import torch
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt

r0 = 1
r1 = 6
r2 = 7
rate = 0.5

class ToyDataset(IterableDataset):
    MIN = -7
    MAX = 7

    DIMENTION = 2

    def __iter__(self):
        while True:
            choice = random.random()
            if(choice < 0.5):
                radius = random.random()*(r2-r1) + r1
                angle = random.random() * math.pi
                x1 = radius * math.cos(angle)
                x2 = radius * math.sin(angle)
                x = torch.tensor([x1, x2])
                y = torch.tensor(1)
            else:
                radius = random.random()*r0
                angle = random.random() * 2 * math.pi
                x1 = radius * math.cos(angle)
                x2 = radius * math.sin(angle)
                x = torch.tensor([x1, x2])
                y = torch.tensor(0)
            yield x, y

    @classmethod
    def plot_range(cls):
        padding = (cls.MAX - cls.MIN) * 0.05
        return cls.MIN - padding, cls.MAX + padding


def visualize_dataset():
    dataloader = DataLoader(ToyDataset(), batch_size=1024)
    x, y = next(iter(dataloader))
    plt.xlim(ToyDataset.plot_range())
    plt.ylim(ToyDataset.plot_range())
    a_examples = x[y == 0]
    b_examples = x[y == 1]
    plt.plot(a_examples[:, 0], a_examples[:, 1], 'o', label='a')
    plt.plot(b_examples[:, 0], b_examples[:, 1], 'o', label='b')
    plt.title('Toy Dataset')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/toy_dataset.png')

if __name__ == "__main__":
    visualize_dataset()
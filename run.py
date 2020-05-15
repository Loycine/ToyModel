import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim as optim
import matplotlib.pyplot as plt

from toy_dataset import ToyDataset
from toy_model import ToyNet

logger = logging.getLogger(__name__)

CHECKPOINT_PER_STEP = 200
STOP_PATIENCE = 10
SAVE_PATH = Path('./saved_models')


def train(name, model, dataset, optimizer):
    logger.info('=====================')
    logger.info('Training model %s', name)
    logger.info('=====================')

    dataloader = DataLoader(dataset, batch_size=256)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    stop_countdown = float('inf')

    for step, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

        if step % CHECKPOINT_PER_STEP == 0:
            acc = evaluate(model, dataset)
            if acc > best_acc:
                best_acc = acc
                stop_countdown = STOP_PATIENCE
                save_path = SAVE_PATH / f'{name}_best'
                torch.save(model.state_dict(), save_path)
                logger.info('Saved new model to %s', save_path)
            else:
                stop_countdown -= 1
            logger.info('Step: %d; Count down: %d; Best accuracy: %f', step, stop_countdown, best_acc)

        if stop_countdown <= 0 or best_acc >= 1.0:
            break
    logger.info('Train finished')


EVAL_BATCH_SIZE = 4096


def evaluate(model, dataset):
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    x, y = next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        prediction = model(x).argmax(dim=-1)
        acc = (prediction == y).sum().item() / EVAL_BATCH_SIZE
    logger.info('Accuracy: %f', acc)
    return acc


def train_net():
    model = ToyNet()
    dataset = ToyDataset()
    optimizer = Adam(model.parameters())
    train('net', model, dataset, optimizer)


def visualize_net():
    model = ToyNet()
    model.load_state_dict(torch.load(SAVE_PATH / 'net_best'))
    model.eval()

    RESOLUTION = 500
    plot_range = ToyDataset.plot_range()
    
    x1 = torch.linspace(*plot_range, steps=RESOLUTION)
    x2 = torch.linspace(*plot_range, steps=RESOLUTION)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2)
    x = torch.stack((grid_x1, grid_x2), dim=-1)
    x = x.view(-1, ToyDataset.DIMENTION)
    with torch.no_grad():
        prediction = model(x).argmax(dim=-1).view(RESOLUTION, RESOLUTION)
    plt.contourf(grid_x1, grid_x2, prediction)

    dataloader = DataLoader(ToyDataset(), batch_size=512)
    x, y = next(iter(dataloader))
    a_examples = x[y == 0]
    b_examples = x[y == 1]
    plt.plot(a_examples[:, 0], a_examples[:, 1], 'o', label='a')
    plt.plot(b_examples[:, 0], b_examples[:, 1], 'o', label='b')
    plt.title('Net Trained to converge')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/net.png')


def main():
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    train_net()
    visualize_net()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

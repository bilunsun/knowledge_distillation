from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_train_test(batch_size=64):
    mnist_dataset = datasets.MNIST(".data", download=True, transform=transforms.ToTensor())

    train_dataset, test_dataset = random_split(mnist_dataset, [55_000, 5_000])

    return DataLoader(train_dataset, batch_size), DataLoader(test_dataset, batch_size)
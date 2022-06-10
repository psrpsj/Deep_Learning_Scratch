from torchvision import datasets
import numpy as np


def load_mnist(normalize=True, flatten=True):
    # MNIST dataset
    mnist_train = datasets.MNIST(root="./data/", train=True, download=True)
    mnist_test = datasets.MNIST(root="./data/", train=False, download=True)
    print("mnist_train:\n", mnist_train, "\n")
    print("mnist_test:\n", mnist_test, "\n")
    print("Done.")

    train_img = []
    train_label = []
    for data in mnist_train:
        train_img.append(np.array(data[0]).reshape(-1, 784))
        train_label.append(data[1])

    test_img = []
    test_label = []
    for test_data in mnist_test:
        test_img.append(np.array(test_data[0]).reshape(-1, 784))
        test_label.append(test_data[1])

    train_img = np.array(train_img)
    train_label = np.array(train_label)
    test_img = np.array(test_img)
    test_label = np.array(test_label)

    if normalize:
        train_img = train_img.astype(np.float32)
        train_img /= 255.0
        test_img = test_img.astype(np.float32)
        test_img /= 255.0

    if not flatten:
        train_img = train_img.reshape(-1, 1, 28, 28)
        test_img = test_img.reshape(-1, 1, 28, 28)

    return train_img, train_label, test_img, test_label

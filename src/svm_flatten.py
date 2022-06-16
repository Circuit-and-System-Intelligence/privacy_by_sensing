#!/usr/bin/env python3

import torch
from torchvision import datasets, transforms
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

frequency = np.array(
    [
        0.00,
        997600000.00,
        2035000000.00,
        3055000000.00,
        4051000000.00,
        5018000000.00,
        5933000000.00,
        6863000000.00,
        7763000000.00,
        8646000000.00,
        9484000000.00,
        10260000000.00,
        10950000000.00,
        11720000000.00,
        12460000000.00,
        12890000000.00,
    ]
)


def nonlinearity(x, frequency):
    out = frequency / (frequency[1] * 15)
    if min(x) < 0:
        out_n = -out[1:].flip(0)
        out = np.cat((out_n, out))
        x = x + 15
    return out[x]


# Load the training data
def MNIST_DATASET_TRAIN(train_amount, downloads=False):
    training_data = datasets.MNIST(
        "../data", train=True, transform=transforms.ToTensor(), download=downloads
    )

    train_data = training_data.data.numpy()[:train_amount]
    train_labels = training_data.targets.numpy()[:train_amount]
    print("Training data size: ", train_data.shape)
    print("Training data label size: ", train_labels.shape)
    # plt.imshow(train_data[0])
    # plt.show()

    train_data = train_data / 255.0

    return train_data, train_labels


# Load the testing data
def MNIST_DATASET_TEST(test_amount, downloads=False):
    testing_data = datasets.MNIST(
        "../data", train=False, transform=transforms.ToTensor(), download=downloads
    )

    test_data = testing_data.data.numpy()[:test_amount]
    test_labels = testing_data.targets.numpy()[:test_amount]
    print("Testing data size: ", test_data.shape)
    print("Testing data label size: ", test_labels.shape)

    test_data = test_data / 255.0

    return test_data, test_labels


# Train the model
def train(model, train_data, train_label, args):
    model.fit(train_data, train_label)
    score = model.score(train_data, train_label)
    print("Score: ", score)

    cv_scores = cross_val_score(model, train_features, train_label, cv=10)
    print("CV average score: %.2f" % cv_scores.mean())
    return model


def generate_projection_matrix(k, nonlinear_mult=False):
    random_projection_matrix = np.random.normal(0, 1 / np.sqrt(k), size=(28 * 28, k))
    print("l2 sensitivty", lp_sensitivity(random_projection_matrix, 2))
    random_projection_matrix = (
        np.digitize(random_projection_matrix, np.linspace(-1, 1, 32)) - 16
    )
    if nonlinear_mult:
        random_projection_matrix = nonlinearity(random_projection_matrix, mode=1)

    random_projection_matrix = random_projection_matrix / 16.0
    print("l2 sensitivty after digitize", lp_sensitivity(random_projection_matrix, 2))
    return random_projection_matrix


def lp_sensitivity(matrix, lp):
    sensitivity = abs(matrix) ** lp
    sensitivity = np.sqrt(sensitivity.sum(1))
    sensitivity = sensitivity.max()
    return sensitivity


def generate_additive_noise(k, sigma, size, args):
    return np.random.normal(0, sigma, size=(size, k))


if __name__ == "__main__":
    # Training Arguments Settings
    parser = argparse.ArgumentParser(description="SVM")
    parser.add_argument(
        "--download_MNIST",
        default=True,
        metavar="DL",
        help="Download MNIST (default: True)",
    )
    parser.add_argument(
        "--train_amount", type=int, default=60000, help="Amount of training samples"
    )
    parser.add_argument(
        "--test_amount", type=int, default=2000, help="Amount of testing samples"
    )
    parser.add_argument(
        "--projection_dimension",
        type=int,
        default=80,
        help="Dimensions after projection",
    )
    parser.add_argument(
        "--nonlinear_mult", default=False, help="Add nonlinear multiplication effects",
    )
    args = parser.parse_args()

    # Print Arguments
    print("\n----------Argument Values-----------")
    for name, value in vars(args).items():
        print("%s: %s" % (str(name), str(value)))
    print("------------------------------------\n")

    train_data, train_label = MNIST_DATASET_TRAIN(
        args.train_amount, downloads=args.download_MNIST
    )
    test_data, test_label = MNIST_DATASET_TEST(
        args.test_amount, downloads=args.download_MNIST
    )

    train_features = train_data.reshape(args.train_amount, -1)
    test_features = test_data.reshape(args.test_amount, -1)

    rnd_matrix = generate_projection_matrix(args.projection_dimension, args)
    train_features = np.matmul(train_features, rnd_matrix)
    test_features = np.matmul(test_features, rnd_matrix)

    # Train SVM
    print("-----Training SVM-----")
    lsvc = LinearSVC(dual=False)

    lsvc = train(lsvc, train_features, train_label, args)

    print("-----Testing SVM------")
    pred_label = lsvc.predict(test_features)

    cm = confusion_matrix(test_label, pred_label)
    print(cm)

    cr = classification_report(test_label, pred_label)
    print(cr)

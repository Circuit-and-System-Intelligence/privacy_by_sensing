#!/usr/bin/env python3

import torch
from torchvision import datasets, transforms
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import numpy as np


# Transform the array x according
def nonlinearity(x, frequency):
    if frequency is None:
        frequency = np.array(
            [
                0.46222,
                0.58199,
                0.71311,
                0.86543,
                1.02708,
                1.19686,
                1.36171,
                1.54039,
                1.72681,
                1.89204,
                2.05703,
                2.21486,
                2.36614,
                2.52254,
                2.66734,
                2.77935,
            ]
        )
    out = frequency / (frequency[0] * 16)
    if np.min(x) < 0:
        out_n = np.flip(-out, 0)
        out = np.concatenate((out_n, out))
        x = x + 16
    return out[x]


# Load the training data
def MNIST_DATASET_TRAIN(train_amount, downloads=False, nonlinear_mult=False):
    training_data = datasets.MNIST(
        "../data", train=True, transform=transforms.ToTensor(), download=downloads
    )

    train_data = training_data.data.numpy()[:train_amount]
    train_labels = training_data.targets.numpy()[:train_amount]
    print("Training data size: ", train_data.shape)
    print("Training data label size: ", train_labels.shape)
    # plt.imshow(train_data[0])
    # plt.show()

    if nonlinear_mult:
        train_data = nonlinearity(train_data // 16, None)
    else:
        train_data = train_data // 16
        train_data = train_data / 16

    return train_data, train_labels


# Load the testing data
def MNIST_DATASET_TEST(test_amount, downloads=False, nonlinear_mult=False):
    testing_data = datasets.MNIST(
        "../data", train=False, transform=transforms.ToTensor(), download=downloads
    )

    test_data = testing_data.data.numpy()[:test_amount]
    test_labels = testing_data.targets.numpy()[:test_amount]
    print("Testing data size: ", test_data.shape)
    print("Testing data label size: ", test_labels.shape)

    if nonlinear_mult:
        test_data = nonlinearity(test_data // 16, None)
    else:
        test_data = test_data // 16
        test_data = test_data / 16

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
        random_projection_matrix = nonlinearity(random_projection_matrix, None)
    else:
        random_projection_matrix = random_projection_matrix / 16.0

    print("l2 sensitivty after digitize", lp_sensitivity(random_projection_matrix, 2))
    return random_projection_matrix


def lp_sensitivity(matrix, lp):
    sensitivity = abs(matrix) ** lp
    sensitivity = np.sqrt(sensitivity.sum(1))
    sensitivity = sensitivity.max()
    return sensitivity


def generate_additive_noise(k, sigma, samples):
    return np.random.normal(0, sigma, size=(samples, k))


def minimum_sigma(delta, epsilon, w_2):
    return w_2 * np.sqrt(2 * (np.log(1 / (2 * delta)) + epsilon)) / epsilon


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
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="The amount of additive noise added to test samples",
    )
    parser.add_argument(
        "--test_name", help="Name for the result files",
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

    rnd_matrix = generate_projection_matrix(
        args.projection_dimension, args.nonlinear_mult
    )
    train_features = np.matmul(train_features, rnd_matrix)
    test_features = np.matmul(test_features, rnd_matrix)
    noise_matrix = generate_additive_noise(args.projection_dimension, args.sigma, args.test_amount)
    test_features = test_features + noise_matrix

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

    score = precision_score(test_label, pred_label, average="macro")
    print(score)

    name = args.test_name + "_results"
    with open(name, "a", encoding="utf-8") as f:
        f.write(f"{score}\n")

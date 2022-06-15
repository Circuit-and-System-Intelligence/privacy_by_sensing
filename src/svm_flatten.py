#!/usr/bin/env python3

import torch
from torch import nn
from torchvision import datasets, transforms
import argparse
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


def nonlinearity(x, mode=0):
    if mode == 0:
        x = x[0].to(torch.long)
    freqency = torch.tensor(
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
    out = freqency / (freqency[1] * 15)
    if mode == 1:
        out_n = -out[1:].flip(0)
        out = torch.cat((out_n, out))
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
    plt.imshow(train_data[0])
    plt.show()

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
    plt.imshow(test_data[0])
    plt.show()

    test_data = test_data / 255.0

    return test_data, test_labels


def train(args, model, device, train_loader, optimizer, epoch, RP):
    # avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0
    l2error = []
    sigma_sum = []
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28 * 28)
        dista = nn.functional.pdist(images)
        dista = dista[dista.nonzero()]
        a1 = dista.mean() ** 2
        sigma = (l2 / dista.mean()) * np.sqrt(np.log(1 / delta)) / epsilon
        sigma_sum += [sigma]
        # if i == 0:
        # sns.displot(image.flatten(), bins=16)
        # plt.savefig("image-hist.png")
        # sum = images[0].matmul(torch.abs(RP)).sum()
        images = images.matmul(RP)
        distb = nn.functional.pdist(images)
        distb = distb[distb.nonzero()]
        a2 = distb.mean() ** 2
        l2error += [((a1 - a2) ** 2) / (a1 ** 2)]
        images = images.reshape(-1, k)

        # Forward pass
        outputs = svm_model(images)
        loss_svm = svm_loss_criteria(outputs, labels)

        # Backward and optimize
        svm_optimizer.zero_grad()
        loss_svm.backward()
        svm_optimizer.step()

        # print("Model's parameter after the update:")
        # for param2 in svm_model.parameters():
        #   print(param2)
        total_batches += 1
        batch_loss += loss_svm.item()

    avg_loss_epoch = batch_loss / total_batches
    l2error = np.mean(l2error)
    sigma_sum = np.mean(sigma_sum)
    print(
        "Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]".format(
            epoch + 1, num_epochs, epoch + 1, avg_loss_epoch
        )
    )


def main(k, epsilon):
    batch_size = 256
    num_classes = 10
    learning_rate = 0.01

    RP = torch.tensor(
        # gs.gram_schmidt(np.random.normal(0, 1 / k, size=(28 * 28, k)))[0],
        np.random.normal(0, 1 / np.sqrt(k), size=(28 * 28, k)),
        dtype=torch.float,
    )
    RP = torch.bucketize(RP, torch.linspace(-1, 1, steps=32)) - 16
    # RP = RP / 16.0
    RP = nonlinearity(RP, mode=1)

    l2 = RP ** 2

    l2 = l2.sum(1).sqrt().max()

    delta = 0.1
    # SVM regression model and Loss
    svm_model = nn.Linear(k, num_classes)
    # model = LogisticRegression(input_size,num_classes)

    # # Loss criteria and SGD optimizer
    # svm_loss_criteria = nn.MultiLabelMarginLoss()
    svm_loss_criteria = nn.CrossEntropyLoss()

    svm_optimizer = torch.optim.Adam(svm_model.parameters(), lr=learning_rate)

    print("l2error", l2error)
    print("sigma", sigma_sum)
    result = [k, epsilon]
    result += [l2error, sigma_sum]
    # Test the SVM Model
    correct = 0.0
    total = 0.0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        dist = nn.functional.pdist(images)
        dist = dist[dist.nonzero()]
        images = images.matmul(RP)
        sigma = (l2 / dist.mean()) * np.sqrt(np.log(1 / delta)) / epsilon
        AN = torch.tensor(
            np.random.normal(0, sigma, size=(len(images), k)), dtype=torch.float
        )
        images = images.add(AN)
        images = images.reshape(-1, k)

        outputs = svm_model(images)
        predicted = outputs.data.max(1, keepdim=True)[1]
        total += labels.size(0)
        correct += (predicted.view(-1).long() == labels).sum()

    accuracy = 100 * (correct.float() / total)
    print(f"{accuracy:.2f}")
    result += [accuracy]
    with open("svm_full.csv", "ab") as f:
        np.savetxt(f, result, newline=", ", fmt="%.2f")
        f.write(b"\n")
    return result


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

    print("Training features", train_features.shape)

    # Train SVM
    print("-----Training SVM-----")
    lsvc = LinearSVC(dual=False)
    print(lsvc)

    lsvc.fit(train_features, train_label)
    score = lsvc.score(train_features, train_label)
    print("Score: ", score)

    cv_scores = cross_val_score(lsvc, train_features, train_label, cv=10)
    print("CV average score: %.2f" % cv_scores.mean())

    pred_label = lsvc.predict(test_features)

    cm = confusion_matrix(test_label, pred_label)
    print(cm)

    cr = classification_report(test_label, pred_label)
    print(cr)

# results = []
# for k in [40, 60, 80, 100, 120, 160, 180]:
#     for e in [0.2, 0.4, 0.6, 0.8, 1]:
#         for i in range(200):
#             print(e, i)
#             measure = main(k, e)
#             results.append([k, e, measure])

#!/usr/bin/env python3

# Copied from https://github.com/AnupamMicrosoft/PyTorch-Classification/blob/master/Linear%20Support%20Vector%20Machines.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import gram_schmidt as gs
import numpy as np


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


def main(k, epsilon):
    batch_size = 256
    num_classes = 10
    num_epochs = 10
    learning_rate = 0.01
    train_data = datasets.MNIST(
        "../original",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: transforms.functional.posterize(x, 4)),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x / 16),
                transforms.Lambda(lambda x: nonlinearity(x)),
            ]
        ),
    )
    test_data = datasets.MNIST(
        "../original",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: transforms.functional.posterize(x, 4)),
                transforms.PILToTensor(),
                transforms.Lambda(lambda x: x / 16),
                transforms.Lambda(lambda x: nonlinearity(x)),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False
    )
    torch.manual_seed(5)

    RP = torch.tensor(
        # gs.gram_schmidt(np.random.normal(0, 1 / k, size=(28 * 28, k)))[0],
        np.random.normal(0, 1 / np.sqrt(k), size=(28*28, k)),
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

    for epoch in range(num_epochs):
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
        np.savetxt(f, result, newline=', ', fmt='%.2f')
        f.write(b"\n")
    return result


if __name__ == "__main__":
    results = []
    for k in [40, 60, 80, 100, 120, 160, 180]:
        for e in [0.2, 0.4, 0.6, 0.8, 1]:
            for i in range(200):
                print(e, i)
                measure = main(k, e)
                results.append([k, e, measure])

    print(results)

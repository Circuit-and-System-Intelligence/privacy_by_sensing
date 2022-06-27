#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def getresult(name, n):
    name = f"{n}.{name}_results"
    return np.genfromtxt(name)


def generate_plot(name, x_values):
    data = []

    for i in range(1, 51):
        r = getresult(name, i)
        data.append(r)

    data = np.array(data)

    average = np.mean(data, 0)

    std = np.std(data, 0)

    np.savetxt(name + "_raw.csv", data, delimiter=",")
    np.savetxt(name + "_avg.csv", average, delimiter=",")
    np.savetxt(name + "_std.csv", std, delimiter=",")

    plt.figure()
    plt.errorbar(x_values, average, yerr=std, capsize=5, fmt="k")
    plt.xlabel(name)
    plt.ylabel("accuracy")
    plt.savefig(name + ".pdf", dpi=600)
    # plt.show()


x_values1 = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
x_values2 = [80, 100, 120, 140, 160, 180, 200]
generate_plot("sigma", x_values1)
generate_plot("dimension", x_values2)

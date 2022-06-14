import torch
from torchvision import datasets, transforms
import gram_schmidt as gs
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import pdist
import seaborn as sns


def nonlinearity(x, mode=0):
    freqency = np.array(
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
    out = freqency / freqency[1]*15
    if mode == 1:
        out_n = np.flip(-out[1:], 0)
        out = np.concatenate((out_n, out))
        x = x + 15
    return out[x]


def run(k, epsilon, data, labels):
    RP = np.random.normal(0, 1 / np.sqrt(k), size=(28 * 28, k))

    RP = np.digitize(RP, np.linspace(-1, 1, 32)) - 16
    # RP = RP / 16.0

    RP = nonlinearity(RP, mode=1)

    l2 = RP ** 2

    l2 = np.sqrt(l2.sum(1)).max()

    delta = 0.1

    subindices = (
        (labels.to_numpy(dtype=np.uint8) == 0) + (labels.to_numpy(dtype=np.uint8) == 1)
    ).nonzero()[0]
    data = data.to_numpy(dtype=np.uint8)[subindices]
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = np.digitize(data, np.linspace(0, 1, 16)) - 1
    data = nonlinearity(data)
    labels = labels.to_numpy(dtype=np.uint8)[subindices]
    split = int(0.85 * len(labels))
    train_data = data[0:split]
    test_data = data[split:-1]

    dista = pdist(train_data)
    dista = dista[dista.nonzero()]
    a1 = np.mean(dista) ** 2
    sigma = (l2 / np.mean(dista)) * np.sqrt(np.log(1 / delta)) / epsilon
    print("sigma", sigma)
    train_data = np.matmul(train_data.reshape((split, 28 * 28)), RP)
    train_data = train_data.reshape((split, k))
    distb = pdist(train_data)
    distb = distb[distb.nonzero()]
    a2 = np.mean(distb) ** 2
    l2error = ((a1 - a2) ** 2) / (a1 ** 2)
    print("l2error", l2error)
    test_data = np.matmul(test_data.reshape((len(labels) - split - 1, 28 * 28)), RP)
    test_data = test_data.reshape((len(labels) - split - 1, k))
    AN = np.random.normal(0, sigma, size=(len(labels) - split - 1, k))
    test_data = test_data + AN
    n_digits = np.unique(labels).size
    train_labels = labels[0:split]
    test_labels = labels[split:-1]
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    bench_result = bench_k_means(
        kmeans=kmeans,
        name="k-means++",
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
    )

    with open("kmeans.csv", "ab") as f:
        np.savetxt(f, [sigma, l2error, *bench_result], newline=", ", fmt="%.2f")
        f.write(b"\n")

    return [sigma, l2error, *bench_result]


def bench_k_means(kmeans, name, train_data, train_labels, test_data, test_labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    estimator = make_pipeline(StandardScaler(), kmeans).fit(train_data)
    results = [name, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(train_labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            train_data, estimator[-1].labels_, metric="euclidean", sample_size=300,
        )
    ]

    same = 0
    diff = 0
    predicted = estimator.predict(test_data)
    for i in range(len(test_labels)):
        if test_labels[i] == predicted[i]:
            same = same + 1
        else:
            diff = diff + 1

    correct = same if same > diff else diff

    accuracy = correct / len(test_labels)

    results += [accuracy]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.3f}"
    )
    # print(formatter_result.format(*results))
    print("homogeneity", results[2])
    print("completeness", results[3])
    print("silhouette", results[7])
    print("accuracy", accuracy)
    return [results[2], results[3], results[7], accuracy]


if __name__ == "__main__":
    data, labels = fetch_openml(name="mnist_784", return_X_y=True, cache=True)
    results = []
    for k in [40, 60, 80, 100, 120, 140, 160]:
        for e in [0.2, 0.4, 0.6, 0.8, 1]:
            for i in range(200):
                print("k", k, "e", e)
                measure = run(k, e, data, labels)
                results.append([k, e, measure])

    print(results)

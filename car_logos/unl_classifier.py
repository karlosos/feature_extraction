import numpy as np
import cv2
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from unl import unl_fourier


class UNLFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, size=5):
        self.classes_ = None
        self.template_dict_ = None
        self.size = size

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        unlf_descriptors = []
        labels = []

        for i in range(len(X)):
            im = cv2.imread(X[i], cv2.IMREAD_GRAYSCALE).astype("uint8")
            unlf_desc = unl_fourier(im, self.size)
            unlf_descriptors.append(unlf_desc)
            labels.append(y[i])

        data = {"unlf": unlf_descriptors, "label": labels}

        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            im = cv2.imread(x, cv2.IMREAD_GRAYSCALE).astype("uint8")
            descriptors = unl_fourier(im, self.size)
            y_pred.append(self.closest_template(descriptors))
        return y_pred

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_["unlf"].tolist()
        distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]["label"]
        return closest_label


def main():
    from simple_shape_descriptors import prepare_dataset

    X_train, y_train, X_test, y_test = prepare_dataset()

    sizes = [5, 20, 30, 35, 40]

    for size in sizes:
        clf = UNLFClassifier(size=size)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_test, y_pred)
        print(f"Size: {size} Acc: {acc}")


def experiment_30():
    from simple_shape_descriptors import prepare_dataset

    X_train, y_train, X_test, y_test = prepare_dataset()
    size = 30

    clf = UNLFClassifier(size=size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, y_pred)
    print(f"Size: {size} Acc: {acc}")

    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    plot_confusion_matrix(clf, X_test, y_test)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    experiment_30()

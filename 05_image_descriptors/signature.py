import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import pandas as pd

from simple_shape_descriptors import object_contour
from simple_shape_descriptors import prepare_dataset


def signature_descriptor(im, length):
    contour = object_contour(im)
    x, y = center_of_contour(im, contour)
    dists = calculate_dists(contour, [x, y])
    downsampled_y, downsampled_x = scale_dists_length(dists, length)

    return downsampled_y, downsampled_x


def center_of_contour(image, contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def calculate_dists(contour, center):
    points = []
    for point in contour:
        points.append((point[0, 0], point[0, 1]))

    points = np.array(points)

    dists = distance.cdist([center], points, "euclidean")[0]
    return dists


def scale_dists_length(dists, length):
    downsampled_x = np.linspace(0, len(dists), length)
    downsampled_y = np.interp(downsampled_x, np.arange(len(dists)), dists)
    return downsampled_y, downsampled_x


class SignatureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, signature_length=200):
        self.classes_ = None
        self.template_dict_ = None
        self.signature_length = signature_length

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        signatures = []
        labels = []

        for i in range(len(X)):
            im = cv2.imread(X[i], cv2.IMREAD_GRAYSCALE).astype("uint8")
            signature, _ = signature_descriptor(im, self.signature_length)
            signatures.append(signature)
            labels.append(y[i])

        data = {"signature": signatures, "label": labels}

        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            im = cv2.imread(x, cv2.IMREAD_GRAYSCALE).astype("uint8")
            descriptor, _ = signature_descriptor(im, self.signature_length)
            y_pred.append(self.closest_template(descriptor))
        return y_pred

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_["signature"].tolist()
        distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]["label"]
        return closest_label


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()
    clf = SignatureClassifier(signature_length=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, y_pred)
    print(f"Acc: {acc}")

    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    plot_confusion_matrix(clf, X_test, y_test)
    plt.xticks(rotation=90)
    plt.show()


def mlp():
    from sklearn.neural_network import MLPClassifier

    # Prepare X_train, X_test
    X_train_names, y_train, X_test_names, y_test = prepare_dataset()
    X_train = []
    X_test = []

    signature_length = 200

    for file_path in X_train_names:
        im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype("uint8")
        signature, _ = signature_descriptor(im, signature_length)
        X_train.append(signature)

    for file_path in X_test_names:
        im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype("uint8")
        signature, _ = signature_descriptor(im, signature_length)
        X_test.append(signature)

    # Create MLP
    clf = MLPClassifier(random_state=1, max_iter=300,
                        hidden_layer_sizes=(100, 50)).fit(X_train, y_train)
    print("MLP score", clf.score(X_test, y_test))

    # Confusion matrix
    from sklearn.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt
    plot_confusion_matrix(clf, X_test, y_test)
    plt.xticks(rotation=90)
    plt.show()


if __name__ == "__main__":
    main()
    mlp()

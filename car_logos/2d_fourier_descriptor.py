import numpy as np
import cv2
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist


class FourierClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, size=5):
        self.classes_ = None
        self.template_dict_ = None
        self.size = size

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        fouriers = []
        labels = []

        for i in range(len(X)):
            im = cv2.imread(X[i], cv2.IMREAD_GRAYSCALE).astype("uint8")
            fourier_desc = self.fourier_desc(im, self.size)
            fouriers.append(fourier_desc)
            labels.append(y[i])

        data = {"fourier": fouriers, "label": labels}

        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            im = cv2.imread(x, cv2.IMREAD_GRAYSCALE).astype("uint8")
            descriptors = self.fourier_desc(im, self.size)
            y_pred.append(self.closest_template(descriptors))
        return y_pred

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_["fourier"].tolist()
        distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]["label"]
        return closest_label

    @staticmethod
    def fourier_desc(img, size):
        img_fft = np.fft.fft2(img)
        spectrum = np.log(1 + np.abs(img_fft))
        out = []
        for i in range(0, size):
            tmp = []
            for j in range(0, size):
                tmp.append(spectrum[i][j])
            out.append(tmp)
        return list(np.concatenate(out).flat)


def main():
    from simple_shape_descriptors import prepare_dataset

    X_train, y_train, X_test, y_test = prepare_dataset()

    sizes = [5, 20, 30, 35, 40]

    for size in sizes:
        clf = FourierClassifier(size=size)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        from sklearn.metrics import accuracy_score

        acc = accuracy_score(y_test, y_pred)
        print(f"Size: {size} Acc: {acc}")


if __name__ == "__main__":
    main()

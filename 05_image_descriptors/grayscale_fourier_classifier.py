import numpy as np
import cv2
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, descriptor, args = {}):
        self.classes_ = None
        self.template_dict_ = None
        self.descriptor_ = descriptor
        self.args_ = args

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        features = []
        labels = []

        for i in range(len(X)):
            image_path = X[i]
            feature = self.descriptor_(image_path, **self.args_)
            features.append(feature)
            labels.append(y[i])

        data = {"feature": features, "label": labels}

        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            features = self.descriptor_(x, **self.args_)
            y_pred.append(self.closest_template(features))
        return y_pred

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_["feature"].tolist()
        distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]["label"]
        return closest_label


def main():
    from grayscale_fourier import grayscale_fourier_desc
    from color_classifier import prepare_dataset

    from sklearn.metrics import accuracy_score
    import pandas as pd

    sizes = [2, 3, 5, 7, 10]
    data = {"size": [], "acc": []}

    X_train, y_train, X_test, y_test = prepare_dataset()

    for size in sizes:
        clf = TemplateClassifier(descriptor=grayscale_fourier_desc, args={"size": size})
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        data["size"].append(size)
        data["acc"].append(acc)
        print(f"{size} => {acc}")

    print()
    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    main()

import numpy as np
import cv2
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, descriptor):
        self.classes_ = None
        self.template_dict_ = None
        self.descriptor_ = descriptor

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        features = []
        labels = []

        for i in range(len(X)):
            image_path = X[i]
            # img = cv2.imread(X[i], cv2.IMREAD_GRAYSCALE).astype("uint8")
            feature = self.descriptor_(image_path)
            features.append(feature)
            labels.append(y[i])

        data = {"feature": features, "label": labels}

        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            features = self.descriptor_(x)
            y_pred.append(self.closest_template(features))
        return y_pred

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_["feature"].tolist()
        distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]["label"]
        return closest_label

FOLDERS = [
    "01_bmw",
    "02_kia",
    "03_mitsubishi",
    "04_volvo",
    "05_peugeout",
    "06_honda",
    "07_subaru",
    "08_tesla",
    "09_renault",
    "10_toyota",
]


def prepare_dataset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for name in FOLDERS:
        files = os.listdir("./dataset/" + name)
        files = [f"./dataset/{name}/{file}" for file in files]
        y = [name for _ in range(len(files))]
        X_train_folder, X_test_folder, y_train_folder, y_test_folder = train_test_split(
            files, y, test_size=len(files) - 2, random_state=1337
        )
        X_train += X_train_folder
        y_train += y_train_folder
        X_test += X_test_folder
        y_test += y_test_folder
    return X_train, y_train, X_test, y_test


def main():
    from color_descriptors import dominant_color_desc
    from color_descriptors import dominant_color_2_desc
    from color_descriptors import color_layout_desc
    from color_descriptors import color_mean_desc
    from color_descriptors import color_hist_rgb_desc
    from color_descriptors import color_hist_hsv_desc
    from color_descriptors import scalable_color_desc 

    from sklearn.metrics import accuracy_score
    import pandas as pd
    import time

    descriptors = [
        # dominant_color_desc,
        dominant_color_2_desc,
        scalable_color_desc,
        color_layout_desc,
        color_mean_desc,
        color_hist_rgb_desc,
        color_hist_hsv_desc
    ]

    data = {'name': [], 'acc': []}

    X_train, y_train, X_test, y_test = prepare_dataset()

    for descriptor in descriptors:
        t1 = time.time()
        clf = TemplateClassifier(descriptor=descriptor)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        name = descriptor.__name__
        data['name'].append(name)
        data['acc'].append(acc)
        print(f"{name} => {acc} \t {time.time() - t1}")

    print()
    df = pd.DataFrame(data)
    print(df)


if __name__ == "__main__":
    main()

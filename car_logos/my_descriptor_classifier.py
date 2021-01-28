import numpy as np
import cv2
import os
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, descriptor, args={}):
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
    from my_descriptor import my_desc
    from color_classifier import prepare_dataset

    from sklearn.metrics import accuracy_score
    import pandas as pd

    from rich.console import Console

    console = Console()

    sizes = [8, 16, 32, 64, 96, 128, 256]
    data = {"size": [], "acc": []}

    X_train, y_train, X_test, y_test = prepare_dataset()

    for size in sizes:
        with console.status(
            f"[bold green] Training classifier with size {size}..."
        ) as _:
            clf = TemplateClassifier(descriptor=my_desc, args={"size": size})
            clf.fit(X_train, y_train)
        with console.status(f"[bold red] Testing classifier with size {size}...") as _:
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        data["size"].append(size)
        data["acc"].append(acc)
        print(f"{size} => {acc}")

    print()
    df = pd.DataFrame(data)
    print(df)


def mlp():
    from my_descriptor import my_desc
    from color_classifier import prepare_dataset

    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    from rich.console import Console

    console = Console()

    X_train_names, y_train, X_test_names, y_test = prepare_dataset()
    X_train = []
    X_test = []

    with console.status("[bold green] Calculating descriptors...") as _:
        for img_path in X_train_names:
            desc = my_desc(img_path, 10)
            X_train.append(desc)

        for img_path in X_test_names:
            desc = my_desc(img_path, 10)
            X_test.append(desc)

    with console.status("[bold green] Training classifiers...") as _:
        clf_mlp = MLPClassifier(
            random_state=1, max_iter=300, hidden_layer_sizes=(250, 50)
        ).fit(X_train, y_train)
        clf_svm = SVC().fit(X_train, y_train)
        clf_lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    with console.status("[bold red] Testing classifiers...") as _:
        y_pred_mlp = clf_mlp.predict(X_test)
        y_pred_svm = clf_svm.predict(X_test)
        y_pred_lr = clf_lr.predict(X_test)

    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    print(f"Accuracy MLP: {acc_mlp}")
    print(f"Accuracy SVM: {acc_svm}")
    print(f"Accuracy LR: {acc_lr}")


if __name__ == "__main__":
    main()
    # mlp()

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

folders = ["01_bmw", "02_kia", "03_mitsubishi", "04_volvo",
           "05_peugeout", "06_honda", "07_subaru", "08_tesla",
           "09_renault", "10_toyota"]


def prepare_dataset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for name in folders:
        files = os.listdir('./dataset/' + name)
        files = [f"./dataset/{name}/{file}" for file in files]
        y = [name for _ in range(len(files))]
        X_train_folder, X_test_folder, y_train_folder, y_test_folder = train_test_split(files, y,
                                                                                        test_size=len(files) - 2,
                                                                                        random_state=42)
        X_train += X_train_folder
        y_train += y_train_folder
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()


if __name__ == '__main__':
    main()

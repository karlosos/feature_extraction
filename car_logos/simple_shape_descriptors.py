import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.spatial.distance import cdist

folders = ["01_bmw", "02_kia", "03_mitsubishi", "04_volvo",
           "05_peugeout", "06_honda", "07_subaru", "08_tesla",
           "09_renault", "10_toyota"]


def prepare_dataset():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for name in folders:
        files = os.listdir('./dataset/binarized/' + name)
        files = [f"./dataset/binarized/{name}/{file}" for file in files]
        y = [name for _ in range(len(files))]
        X_train_folder, X_test_folder, y_train_folder, y_test_folder = train_test_split(files, y,
                                                                                        test_size=len(files) - 2,
                                                                                        random_state=1337)
        X_train += X_train_folder
        y_train += y_train_folder
        X_test += X_test_folder
        y_test += y_test_folder
    return X_train, y_train, X_test, y_test


def area(contour):
    return cv2.contourArea(contour)


def area2(im):
    return np.sum(im == 0)


def object_contour(im):
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[1]


def perimeter(contour):
    perimeter = cv2.arcLength(contour, True)
    return perimeter


def convex_hull(contour):
    hull = cv2.convexHull(contour)
    return hull


def diameter(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return radius*2


def solidity(contour):
    # https://docs.opencv.org/3.4/da/dc1/tutorial_js_contour_properties.html
    object_area = area(contour)
    hull = convex_hull(contour)
    hull_area = area(hull)
    solidity = object_area/hull_area
    return solidity


def compactness(contour):
    object_area = area(contour)
    object_perimeter = perimeter(contour)
    compactness = object_perimeter**2/object_area
    return compactness


def roundness(contour):
    p = perimeter(contour)
    a = area(contour)
    gamma = p**2 / (4 * np.pi * a)
    return gamma


def eccentricity(contour):
    _, (w, h), _= cv2.minAreaRect(contour)
    return w/h


def draw_countours(X_train):
    for i in range(len(X_train)):
        im = cv2.imread(X_train[i], cv2.IMREAD_GRAYSCALE).astype('uint8')
        ret, thresh = cv2.threshold(im, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        img = cv2.drawContours(im, contours, 1, (255,0,0), 3)
        print(contours[1])
        cv2.imshow("Image", img)
        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()  # destroys the window showing image


def template_dictionary(X_train, y_train):
    areas = []
    perimeters = []
    diameters = []
    solidities = []
    compactnesses = []
    roundnesses = []
    labels = []
    eccentricities = []

    for i in range(len(X_train)):
        im = cv2.imread(X_train[i], cv2.IMREAD_GRAYSCALE).astype('uint8')
        contour = object_contour(im)
        areas.append(area(contour))
        perimeters.append(perimeter(contour))
        diameters.append(diameter(contour))
        solidities.append(solidity(contour))
        compactnesses.append(compactness(contour))
        roundnesses.append(roundness(contour))
        eccentricities.append(eccentricity(contour))
        labels.append(y_train[i])

    data = {'area': areas,
            'perimeter': perimeters,
            'diameter': diameters,
            'solidity': solidities,
            'compactness': compactnesses,
            'roundness': roundnesses,
            'eccentricity': eccentricities,
            'label': labels}
    df = pd.DataFrame.from_dict(data)
    return df


def calculate_shape_descriptors(x):
    im = cv2.imread(x, cv2.IMREAD_GRAYSCALE).astype('uint8')
    contour = object_contour(im)
    descriptors = [area(contour), perimeter(contour), diameter(contour), solidity(contour), compactness(contour),
                   roundness(contour), eccentricity(contour)]
    return descriptors


def closest_template(descriptors, template_dict):
    template_descriptors = template_dict.loc[:, 'area':'eccentricity']
    distances = cdist([descriptors[3:7]], template_descriptors.iloc[:, 3:7]).mean(axis=0)
    closest_label = template_dict.iloc[distances.argmin()]['label']
    return closest_label


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()
    template_dict = template_dictionary(X_train, y_train)

    y_pred = []
    for i in range(len(X_test)):
        x = X_test[i]
        descriptors = calculate_shape_descriptors(x)
        y_pred.append(closest_template(descriptors, template_dict))
    print(y_pred)
    print(y_test)

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    # from sklearn.metrics import plot_confusion_matrix
    # plot_confusion_matrix(mc, X_test, y_test)


if __name__ == '__main__':
    main()

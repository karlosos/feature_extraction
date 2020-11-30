from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import cdist

import simple_shape_descriptors as ssd


class ShapeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, limited=False):
        self.classes_ = None
        self.template_dict_ = None
        self.limited = limited

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        areas = []
        perimeters = []
        diameters = []
        solidities = []
        compactnesses = []
        roundnesses = []
        labels = []
        eccentricities = []

        for i in range(len(X)):
            im = cv2.imread(X[i], cv2.IMREAD_GRAYSCALE).astype('uint8')
            contour = ssd.object_contour(im)
            areas.append(ssd.area(contour))
            perimeters.append(ssd.perimeter(contour))
            diameters.append(ssd.diameter(contour))
            solidities.append(ssd.solidity(contour))
            compactnesses.append(ssd.compactness(contour))
            roundnesses.append(ssd.roundness(contour))
            eccentricities.append(ssd.eccentricity(contour))
            labels.append(y[i])

        data = {'area': areas,
                'perimeter': perimeters,
                'diameter': diameters,
                'solidity': solidities,
                'compactness': compactnesses,
                'roundness': roundnesses,
                'eccentricity': eccentricities,
                'label': labels}
        df = pd.DataFrame.from_dict(data)
        self.template_dict_ = df

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            x = X[i]
            descriptors = self.calculate_shape_descriptors(x)
            y_pred.append(self.closest_template(descriptors))
        return y_pred

    @staticmethod
    def calculate_shape_descriptors(x):
        im = cv2.imread(x, cv2.IMREAD_GRAYSCALE).astype('uint8')
        contour = ssd.object_contour(im)
        descriptors = [ssd.area(contour), ssd.perimeter(contour), ssd.diameter(contour), ssd.solidity(contour), ssd.compactness(contour),
                       ssd.roundness(contour), ssd.eccentricity(contour)]
        return descriptors

    def closest_template(self, descriptors):
        template_descriptors = self.template_dict_.loc[:, 'area':'eccentricity']
        if self.limited:
            distances = cdist([descriptors[3:7]], template_descriptors.iloc[:, 3:7]).mean(axis=0)
        else:
            distances = cdist([descriptors], template_descriptors).mean(axis=0)
        closest_label = self.template_dict_.iloc[distances.argmin()]['label']
        return closest_label

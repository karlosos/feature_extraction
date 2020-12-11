import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from simple_shape_descriptors import object_contour
from simple_shape_descriptors import prepare_dataset


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


def test():
    X_train, y_train, X_test, y_test = prepare_dataset()

    im = cv2.imread(X_train[8], cv2.IMREAD_GRAYSCALE).astype("uint8")
    contour = object_contour(im)

    # plt.imshow(im)
    # plt.show()

    # Center of contour
    x, y = center_of_contour(im, contour)

    # Plot image with center
    cv2.circle(im, (x, y), 7, (255, 255, 255), -1)
    cv2.imshow("Image", im)
    cv2.waitKey(0)

    # Contours, what are those?
    # Contour points start with first upper left point
    # for idx, point in enumerate(contour):
    #    print(point)
    #    cv2.circle(im, (point[0, 0], point[0, 1]), 7, (25 * idx, 25 * idx, 25 * idx), -1)
    #    if idx > 10:
    #       break
    # cv2.imshow("Image", im)
    # cv2.waitKey(0)

    # Calculate distance to every point
    dists = calculate_dists(contour, [x, y])

    # Plot signature
    # print(dists)
    # plt.plot(dists)
    # plt.show()

    # Interpolate dists vector to fixed length e.g. 200 elements
    downsampled_y, downsampled_x = scale_dists_length(dists, 200)

    # Plot points
    plt.plot(dists)
    plt.scatter(downsampled_x, downsampled_y)
    plt.show()

    # Downsampled points length
    print("Downsampled distances lentgh:", downsampled_y.shape)
    print("Distances lentgh:", dists.shape)

    plt.plot(downsampled_y)
    plt.show()


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()
    for idx, X in enumerate(X_train):
        im = cv2.imread(X, cv2.IMREAD_GRAYSCALE).astype("uint8")
        contour = object_contour(im)
        x, y = center_of_contour(im, contour)
        dists = calculate_dists(contour, [x, y])
        downsampled_y, downsampled_x = scale_dists_length(dists, 200)

        # Downsampled points length
        print(idx, X)
        print("Distances lentgh:", dists.shape)
        print("Downsampled distances lentgh:", downsampled_y.shape)


if __name__ == "__main__":
    test()
    # main()

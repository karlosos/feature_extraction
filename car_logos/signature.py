import cv2
import numpy as np

from simple_shape_descriptors import object_contour
from simple_shape_descriptors import prepare_dataset


def center_of_contour(image, contour):
    # compute the center of the contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def test():
    X_train, y_train, X_test, y_test = prepare_dataset()

    im = cv2.imread(X_train[13], cv2.IMREAD_GRAYSCALE).astype("uint8")
    contour = object_contour(im)

    import matplotlib.pyplot as plt

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
    points = []
    for point in contour:
        points.append((point[0, 0], point[0, 1]))

    print(np.array(points))
    points = np.array(points)

    from scipy.spatial import distance

    dists = distance.cdist([[x, y]], points, "euclidean")[0]

    # Plot signature
    # print(dists)
    # plt.plot(dists)
    # plt.show()

    # Interpolate dists vector to fixed length e.g. 200 elements
    print("dists vector:")
    print(dists)
    print(dists.shape)

    output_length = 100
    downsampled_x = np.linspace(0, len(dists), output_length)
    downsampled_y = np.interp(downsampled_x, np.arange(len(dists)), dists)

    # Plot points
    plt.plot(dists)
    plt.scatter(downsampled_x, downsampled_y)
    plt.show()

    # Downsampled points length
    print("Downsampled points lentgh:", downsampled_y.shape)

    plt.plot(downsampled_y)
    plt.show()


if __name__ == "__main__":
    test()

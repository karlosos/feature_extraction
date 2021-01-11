import cv2
import numpy as np
import matplotlib.pyplot as plt

from simple_shape_descriptors import object_contour, prepare_dataset
from signature import center_of_contour, calculate_dists


def unl_whole(img):
    contour = object_contour(img)
    x, y = center_of_contour(img, contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    polar_image = cv2.linearPolar(
        img, center=(x, y), maxRadius=radius, flags=cv2.WARP_POLAR_LINEAR
    )
    polar_image = polar_image.astype(np.uint8)
    scaled_polar_image = cv2.resize(polar_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    result = cv2.rotate(scaled_polar_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
    return result 

def unl(img):
    contour = object_contour(img)
    thresh = cv2.Canny(img, 30, 200)
    x, y = center_of_contour(img, contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    polar_image = cv2.linearPolar(
        thresh, center=(x, y), maxRadius=radius, flags=cv2.WARP_POLAR_LINEAR
    )
    polar_image = polar_image.astype(np.uint8)
    scaled_polar_image = cv2.resize(polar_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    result = cv2.rotate(scaled_polar_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) 
    return result 


def unl_fourier(img, size, whole=False):
    if whole:
        polar_image = unl_whole(img)
    else:
        polar_image = unl(img)

    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))

    result = list(spectrum[:size, :size].flat)
    return result


def main():
    X_train, y_train, X_test, y_test = prepare_dataset()
    name = 3
    file = f"./dataset/tests/{name}.png"
    file = X_train[8]

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype("uint8")
    unl_fourier(img, 4)

    cv2.imshow("Original image", img)
    cv2.imshow("UNL", unl(img))
    cv2.waitKey(0)

    # for i in range(20):
    #     img = cv2.imread(X_train[i], cv2.IMREAD_GRAYSCALE).astype("uint8")
    #     cv2.imshow("Original image", img)
    #     cv2.imshow("UNL", unl(img))
    #     cv2.waitKey(0)


if __name__ == "__main__":
    main()

import numpy as np
import cv2


def grayscale_fourier_desc(filename, size):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.bitwise_not(img)
    # Przekształcenie do skali szarości i do współrzędnych biegunowych
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)
    polar_image = cv2.linearPolar(
        img, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)

    # Przekształcenie fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.bitwise_not(img), cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.bitwise_not(polar_image), cmap="gray")
    # plt.show()

    # Wycinek fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    result = list(spectrum[:size, :size].flat)
    return result

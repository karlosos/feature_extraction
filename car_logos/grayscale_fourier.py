import numpy as np
import cv2


def grayscale_fourier_desc(filename, size):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Przekształcenie do skali szarości i do współrzędnych biegunowych
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    moment = cv2.moments(img)
    x = int(moment["m10"] / moment["m00"])
    y = int(moment["m01"] / moment["m00"])
    centroid = (x, y)
    centroid = (img.shape[0] / 2, img.shape[1] / 2)
    polar_image = cv2.linearPolar(
        img, centroid, max_radius, cv2.WARP_FILL_OUTLIERS
    )
    polar_image = polar_image.astype(np.uint8)
    # polar_image = np.rot90(polar_image)

    # Przekształcenie fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))

    # plt.imshow(spectrum, "gray")
    # plt.title("Spectrum")
    # plt.show()

    # Wycinek fouriera
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    result = list(spectrum[:size, :size].flat)
    return result

